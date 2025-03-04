import sys
import time
import signal
import subprocess
from typing import Dict, Any
from contextlib import contextmanager
from connection import GDBConnectionSystem
from state_save import StateSnapshotSystem
from infer import InferenceSystem

class FirmwareEmulationSystem:
    """
    固件仿真主系统
    集成QEMU仿真、状态捕获、模型推理和状态恢复功能
    """
    def __init__(self, firmware_path: str):
        # 初始化配置
        self.firmware_path = firmware_path
        self.qemu_process = None
        self.gdb_connection = None
        self.snapshot_system = StateSnapshotSystem()
        self.inference_system = InferenceSystem()
        
        # 性能监控
        self.stats = {
            'total_requests': 0,
            'successful_inferences': 0,
            'state_restores': 0
        }

    @contextmanager
    def _qemu_context(self):
        """QEMU进程上下文管理"""
        try:
            self.qemu_process = subprocess.Popen(
                [
                    'qemu-system-x86_64',

                    # '-M', 'versatilepb',
                    # '-kernel', '',
                    '-drive file', self.firmware_path, ',format=raw'
                    '-device', 'e1000'
                    '-S', '-gdb', 'tcp::1234',
                    '-nographic'
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            time.sleep(2)  # 等待QEMU初始化
            yield
        finally:
            if self.qemu_process:
                self.qemu_process.terminate()
                try:
                    self.qemu_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.qemu_process.kill()
                print("QEMU process terminated.")

    def _setup_interrupt_handlers(self):
        """配置信号处理"""
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)

    def _graceful_shutdown(self, signum, frame):
        """优雅关闭系统"""
        print("\nInitiating shutdown sequence...")
        if self.gdb_connection:
            self.gdb_connection.shutdown()
        sys.exit(0)

    def _collect_context(self, gdb_process) -> Dict[str, Any]:
        """收集完整上下文信息"""
        return {
            "registers": self._get_register_state(gdb_process),
            "memory_map": self._get_memory_map(gdb_process),
            "call_stack": self._get_call_stack(gdb_process),
            "interrupt_state": self._get_interrupt_state(gdb_process),
            "peripheral_access": self._get_peripheral_access_info(gdb_process)
        }

    def _get_register_state(self, gdb_process) -> Dict[str, int]:
        """获取寄存器状态"""
        gdb_process.sendline("info registers")
        gdb_process.expect_exact("(gdb)")
        return self._parse_register_output(gdb_process.before.decode())

    def _get_memory_map(self, gdb_process) -> Dict[str, Any]:
        """获取内存映射信息"""
        gdb_process.sendline("info proc mappings")
        gdb_process.expect_exact("(gdb)")
        return self._parse_memory_mappings(gdb_process.before.decode())

    def _get_call_stack(self, gdb_process) -> list:
        """获取函数调用链"""
        gdb_process.sendline("bt")
        gdb_process.expect_exact("(gdb)")
        return self._parse_call_stack(gdb_process.before.decode())

    def _get_interrupt_state(self, gdb_process) -> Dict[str, Any]:
        """获取中断状态"""
        gdb_process.sendline("monitor info irq")
        gdb_process.expect_exact("(gdb)")
        return self._parse_interrupt_info(gdb_process.before.decode())

    def _get_peripheral_access_info(self, gdb_process) -> Dict[str, Any]:
        """获取外设访问信息"""
        gdb_process.sendline("x/4i $pc")
        gdb_process.expect_exact("(gdb)")
        return self._parse_disassembly(gdb_process.before.decode())

    def _parse_register_output(self, output: str) -> Dict[str, int]:
        """解析寄存器输出"""
        registers = {}
        for line in output.splitlines():
            if ' ' in line:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].strip()
                    try:
                        value = int(parts[1], 16)
                        registers[name] = value
                    except ValueError:
                        continue
        return registers

    def _parse_memory_mappings(self, output: str) -> Dict[str, Any]:
        """解析内存映射信息"""
        mappings = []
        for line in output.splitlines()[4:]:  # 跳过标题行
            parts = line.split()
            if len(parts) >= 5:
                mappings.append({
                    "start": int(parts[0], 16),
                    "end": int(parts[1], 16),
                    "size": int(parts[2], 16),
                    "offset": int(parts[3], 16),
                    "objfile": parts[4]
                })
        return mappings

    def _parse_call_stack(self, output: str) -> list:
        """解析调用栈"""
        stack = []
        for line in output.splitlines():
            if '#' in line:
                parts = line.split('#')
                if len(parts) >= 2:
                    frame_info = parts[1].strip()
                    stack.append(frame_info)
        return stack

    def _parse_interrupt_info(self, output: str) -> Dict[str, Any]:
        """解析中断信息"""
        irq_info = {}
        for line in output.splitlines():
            if 'IRQ' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    irq_num = int(parts[0].split()[-1])
                    state = parts[1].strip()
                    irq_info[f"IRQ{irq_num}"] = state
        return irq_info

    def _parse_disassembly(self, output: str) -> Dict[str, Any]:
        """解析反汇编输出"""
        instructions = []
        for line in output.splitlines():
            if ':' in line:
                parts = line.split(':', 1)
                address = int(parts[0].strip(), 16)
                instruction = parts[1].strip()
                instructions.append({
                    "address": address,
                    "instruction": instruction
                })
        return {
            "pc": instructions[0]["address"] if instructions else 0,
            "instructions": instructions
        }

    def restore_snapshot(self, gdb_process, snapshot_id: str):
        """恢复先前的状态快照"""
        snapshot = self.snapshot_system.load_snapshot(snapshot_id)
        if snapshot:
            print(f"Restoring snapshot {snapshot_id}...")
            # 恢复寄存器、内存等信息
            for register, value in snapshot['registers'].items():
                gdb_process.sendline(f"set $ {register} = {value}")
            # 其他恢复步骤可以继续扩展，如恢复内存等
        else:
            print(f"Snapshot {snapshot_id} not found. Cannot restore.")

    def run(self):
        """主运行循环"""
        self._setup_interrupt_handlers()
        
        with self._qemu_context():
            # 初始化GDB连接
            self.gdb_connection = GDBConnectionSystem()
            
            try:
                while True:
                    # 监控硬件观察点
                    if self.gdb_connection.state_queue.empty():
                        time.sleep(0.1)
                        continue
                    
                    # 获取状态数据
                    state_data = self.gdb_connection.state_queue.get()
                    self.stats['total_requests'] += 1
                    
                    # 收集完整上下文
                    full_context = self._collect_context(
                        self.gdb_connection.gdb_process
                    )
                    full_context.update(state_data['context'])
                    
                    try:
                        # 执行模型推理
                        response = self.inference_system.infer_response(
                            full_context,
                            attempt=state_data.get('attempt', 0)
                        )
                        
                        # 应用推理结果
                        self.gdb_connection.response_queue.put(response)
                        self.stats['successful_inferences'] += 1
                        
                    except Exception as e:
                        print(f"Inference failed: {str(e)}")
                        # 恢复状态并重试
                        self.restore_snapshot(
                            self.gdb_connection.gdb_process,
                            state_data['snapshot_id']
                        )
                        self.stats['state_restores'] += 1
                        
                        # 重试逻辑
                        state_data['attempt'] = state_data.get('attempt', 0) + 1
                        if state_data['attempt'] < 3:
                            self.gdb_connection.state_queue.put(state_data)
                        else:
                            print("Maximum retries exceeded")
                            
                    # 打印性能统计
                    if self.stats['total_requests'] % 10 == 0:
                        print("\n=== System Statistics ===")
                        print(f"Total Requests: {self.stats['total_requests']}")
                        print(f"Success Rate: {self.stats['successful_inferences']/self.stats['total_requests']:.2%}")
                        print(f"State Restores: {self.stats['state_restores']}\n")
                        
            except KeyboardInterrupt:
                self._graceful_shutdown(None, None)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <firmware_path>")
        sys.exit(1)
        
    emulator = FirmwareEmulationSystem(sys.argv[1])
    emulator.run()
