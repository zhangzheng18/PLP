import pexpect
import threading
import queue
import time
from typing import Dict, Any
from infer import infer_response
# connection.py
from state_save import (
    save_snapshot,
    restore_snapshot,
    get_snapshot_list,
    get_snapshot_info,
    get_performance_metrics
)

def example_usage():
    class MockGDB:
        def sendline(self, cmd): pass
        def expect_exact(self, *args): pass
    
    gdb = MockGDB()
    
    # 创建快照
    sid, checksum = save_snapshot(gdb)
    
    # 获取快照列表
    print("Available snapshots:", get_snapshot_list())
    
    # 获取详细信息
    print("Snapshot info:", get_snapshot_info(sid))
    
    # 恢复快照
    restore_snapshot(gdb, sid)
    
    # 查看性能指标
    print("Performance:", get_performance_metrics())

class GDBConnectionSystem:
    """
    高可靠GDB连接管理系统
    支持异步通信、状态监控和自动恢复
    """
    def __init__(self, gdb_port=1234, max_retries=3):
        # 连接配置
        self.gdb_port = gdb_port
        self.max_retries = max_retries
        self.connection_lock = threading.RLock()
        
        # 状态管理
        self.current_state = {}
        self.state_queue = queue.Queue(maxsize=10)
        self.response_queue = queue.Queue(maxsize=10)
        
        # 监控线程
        self.monitor_thread = threading.Thread(
            target=self._state_monitor,
            daemon=True
        )
        self.processor_thread = threading.Thread(
            target=self._response_processor,
            daemon=True
        )
        
        # 初始化连接
        self._init_gdb_connection()

    def _init_gdb_connection(self):
        """初始化GDB连接"""
        with self.connection_lock:
            self.gdb_process = pexpect.spawn("gdb", timeout=None)
            self.gdb_process.expect_exact("(gdb)")
            
            # 配置远程连接
            self.gdb_process.sendline(f"target remote :{self.gdb_port}")
            self.gdb_process.expect_exact("(gdb)")
            
            # 设置硬件观察点
            self.gdb_process.sendline("watch *(volatile int*)0x40000000")
            self.gdb_process.expect_exact("(gdb)")
            
            # 启动监控线程
            self.monitor_thread.start()
            self.processor_thread.start()

    def _state_monitor(self):
        """状态监控线程"""
        retry_count = 0
        while True:
            try:
                # 等待状态变化
                self.gdb_process.sendline("continue")
                index = self.gdb_process.expect([
                    pexpect.TIMEOUT, 
                    "Hardware watchpoint",
                    "(gdb)"
                ], timeout=10)
                
                if index == 1:  # 观察点触发
                    # 提取上下文信息
                    context = self._extract_context()
                    
                    # 保存当前状态快照
                    snapshot_id = save_snapshot(self.gdb_process)
                    
                    # 将状态放入队列
                    state_data = {
                        'context': context,
                        'snapshot_id': snapshot_id,
                        'timestamp': time.time()
                    }
                    self.state_queue.put(state_data)
                    
                    # 等待推理结果
                    response = self.response_queue.get(timeout=30)
                    
                    # 应用推理结果
                    self._apply_response(response)
                    
                    # 恢复状态
                    restore_snapshot(self.gdb_process, snapshot_id)
                    
                    retry_count = 0  # 重置重试计数
                    
                elif index == 2:  # GDB提示符
                    continue
                    
            except queue.Empty:
                retry_count += 1
                if retry_count >= self.max_retries:
                    self._reconnect()
                    retry_count = 0
                continue
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
                self._reconnect()

    def _response_processor(self):
        """响应处理线程"""
        while True:
            try:
                # 获取状态数据
                state_data = self.state_queue.get(timeout=5)
                
                # 调用推理系统
                response = infer_response(
                    state_data['context'],
                    attempt=state_data.get('attempt', 0)
                )
                
                # 将响应放入队列
                self.response_queue.put(response)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")
                # 重试逻辑
                state_data['attempt'] = state_data.get('attempt', 0) + 1
                if state_data['attempt'] < self.max_retries:
                    self.state_queue.put(state_data)

    def _extract_context(self) -> Dict[str, Any]:
        """提取当前上下文信息"""
        self.gdb_process.sendline("info registers")
        self.gdb_process.expect_exact("(gdb)")
        register_info = self.gdb_process.before.decode()
        
        self.gdb_process.sendline("x/16x 0x40000000")
        self.gdb_process.expect_exact("(gdb)")
        memory_info = self.gdb_process.before.decode()
        
        return {
            'registers': self._parse_registers(register_info),
            'memory': self._parse_memory(memory_info),
            'timestamp': time.time()
        }

    def _parse_registers(self, register_info: str) -> Dict[str, int]:
        """解析寄存器信息"""
        registers = {}
        for line in register_info.splitlines():
            if ' ' in line:
                name, value = line.split()[:2]
                registers[name.strip()] = int(value, 16)
        return registers

    def _parse_memory(self, memory_info: str) -> Dict[str, int]:
        """解析内存信息"""
        memory = {}
        for line in memory_info.splitlines():
            parts = line.split(':')
            if len(parts) == 2:
                address = parts[0].strip()
                values = [int(x, 16) for x in parts[1].split()]
                memory[address] = values
        return memory

    def _apply_response(self, response: int):
        """应用推理系统返回的响应值"""
        with self.connection_lock:
            self.gdb_process.sendline(f"set *(volatile int*)0x40000000 = {response}")
            self.gdb_process.expect_exact("(gdb)")

    def _reconnect(self):
        """重新建立连接"""
        with self.connection_lock:
            try:
                self.gdb_process.sendline("quit")
                self.gdb_process.expect(pexpect.EOF)
            except:
                pass
            self._init_gdb_connection()

    def shutdown(self):
        """安全关闭系统"""
        with self.connection_lock:
            self.gdb_process.sendline("quit")
            self.gdb_process.expect(pexpect.EOF)
            self.monitor_thread.join(timeout=5)
            self.processor_thread.join(timeout=5)

# 使用示例
if __name__ == "__main__":
    connection_system = GDBConnectionSystem()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        connection_system.shutdown()