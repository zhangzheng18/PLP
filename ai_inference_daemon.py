#!/usr/bin/env python3
"""
AI推断守护进程
监控QEMU的MMIO错误事件，进行AI推断，并将结果写回QEMU系统
"""

import os
import sys
import time
import json
import struct
import mmap
import signal
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import argparse

# 如果有AI推断模块，导入它们
try:
    from ai_register_inference import PeripheralRegisterInference
    HAS_CLOUD_AI = True
except ImportError:
    HAS_CLOUD_AI = False

try:
    from local_ai_inference import LocalPeripheralRegisterInference
    HAS_LOCAL_AI = True
except ImportError:
    HAS_LOCAL_AI = False

# 常量定义
INFERENCE_SHARED_MEM_SIZE = 8192
MAX_INFERRED_REGISTERS = 64

@dataclass
class InferenceRequest:
    fault_addr: int
    fault_pc: int
    fault_size: int
    is_write: bool
    
@dataclass
class RegisterInference:
    offset: int
    value: int
    confidence: int
    size: int
    name: str
    description: str

@dataclass
class InferenceResult:
    device_addr: int
    register_count: int
    registers: List[RegisterInference]
    timestamp: int
    inference_id: int
    need_resume: bool

class InferenceDaemon:
    def __init__(self, bridge_mem_name="/mmio_inference_bridge", 
                 state_mem_name="/mmio_proxy_shared",
                 ai_type="local", api_key=None):
        self.bridge_mem_name = bridge_mem_name
        self.state_mem_name = state_mem_name
        self.ai_type = ai_type
        self.api_key = api_key
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化AI推断系统
        self.ai_inference = None
        self.init_ai_system()
        
        # 共享内存相关
        self.bridge_fd = None
        self.bridge_mem = None
        self.running = True
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def init_ai_system(self):
        """初始化AI推断系统"""
        if self.ai_type == "cloud" and HAS_CLOUD_AI and self.api_key:
            self.logger.info("Initializing cloud AI inference system")
            self.ai_inference = PeripheralRegisterInference(self.api_key)
        elif self.ai_type == "local" and HAS_LOCAL_AI:
            self.logger.info("Initializing local AI inference system")
            self.ai_inference = LocalPeripheralRegisterInference()
        else:
            self.logger.warning("No AI system available, using rule-based inference")
            self.ai_inference = None
    
    def signal_handler(self, signum, frame):
        """信号处理函数"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def open_bridge_memory(self):
        """打开桥接共享内存"""
        try:
            self.bridge_fd = os.open(self.bridge_mem_name, os.O_RDWR)
            self.bridge_mem = mmap.mmap(self.bridge_fd, INFERENCE_SHARED_MEM_SIZE, 
                                      access=mmap.ACCESS_WRITE)
            self.logger.info(f"Bridge memory opened: {self.bridge_mem_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to open bridge memory: {e}")
            return False
    
    def read_bridge_status(self):
        """读取桥接状态"""
        if not self.bridge_mem:
            return None
        
        try:
            # 读取桥接状态结构
            data = self.bridge_mem[:40]  # 基本状态信息
            (pending_inference, inference_complete, fault_addr, fault_pc, 
             fault_size, is_write, qemu_paused, resume_requested) = struct.unpack('IIQQQIII', data)
            
            return {
                'pending_inference': pending_inference,
                'inference_complete': inference_complete,
                'fault_addr': fault_addr,
                'fault_pc': fault_pc,
                'fault_size': fault_size,
                'is_write': bool(is_write),
                'qemu_paused': bool(qemu_paused),
                'resume_requested': bool(resume_requested)
            }
        except Exception as e:
            self.logger.error(f"Failed to read bridge status: {e}")
            return None
    
    def write_inference_result(self, result: InferenceResult):
        """将推断结果写入共享内存"""
        if not self.bridge_mem:
            return False
        
        try:
            # 构建结果数据
            result_data = struct.pack('QIII', 
                                    result.device_addr,
                                    result.register_count,
                                    result.inference_id,
                                    int(result.need_resume))
            
            # 写入寄存器数据
            registers_data = b''
            for reg in result.registers[:MAX_INFERRED_REGISTERS]:
                reg_data = struct.pack('IQIII32s128s',
                                     reg.offset, reg.value, reg.confidence, reg.size,
                                     len(reg.name), reg.name.encode('utf-8')[:32],
                                     reg.description.encode('utf-8')[:128])
                registers_data += reg_data
            
            # 写入到共享内存（跳过状态部分，从结果部分开始）
            result_offset = 40  # 跳过状态信息
            self.bridge_mem[result_offset:result_offset + len(result_data)] = result_data
            self.bridge_mem[result_offset + len(result_data):result_offset + len(result_data) + len(registers_data)] = registers_data
            
            # 标记推断完成
            struct.pack_into('I', self.bridge_mem, 4, 1)  # inference_complete = 1
            struct.pack_into('I', self.bridge_mem, 0, 0)  # pending_inference = 0
            
            self.logger.info(f"Inference result written for device 0x{result.device_addr:x}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write inference result: {e}")
            return False
    
    def perform_rule_based_inference(self, request: InferenceRequest) -> InferenceResult:
        """基于规则的推断（当AI不可用时）"""
        device_addr = request.fault_addr & ~0xFFF  # 对齐到4KB边界
        
        # 简单的规则推断
        registers = []
        
        # 根据地址范围猜测设备类型
        if 0x9000000 <= device_addr <= 0x9010000:
            # 可能是UART设备
            registers = [
                RegisterInference(0x00, 0x0, 90, 4, "UARTDR", "数据寄存器"),
                RegisterInference(0x18, 0x90, 85, 4, "UARTFR", "标志寄存器 - TX ready, RX empty"),
                RegisterInference(0x30, 0x301, 80, 4, "UARTCR", "控制寄存器 - UART/TX/RX enabled"),
            ]
        elif 0x9030000 <= device_addr <= 0x9040000:
            # 可能是GPIO设备
            registers = [
                RegisterInference(0x000, 0x0, 85, 4, "GPIODATA", "数据寄存器"),
                RegisterInference(0x400, 0xFF, 80, 4, "GPIODIR", "方向寄存器 - 全部输出"),
                RegisterInference(0x410, 0x0, 75, 4, "GPIOIE", "中断屏蔽寄存器"),
            ]
        elif 0x10000000 <= device_addr <= 0x20000000:
            # PCIe配置空间范围 - 推断为PCI设备
            registers = [
                RegisterInference(0x00, 0x168c0013, 95, 4, "PCI_VENDOR_ID", "厂商ID: Qualcomm Atheros"),
                RegisterInference(0x04, 0x00000006, 90, 4, "PCI_COMMAND", "命令寄存器 - Bus Master Enable"),
                RegisterInference(0x08, 0x02800000, 85, 4, "PCI_CLASS_REV", "网络控制器类代码"),
                RegisterInference(0x0C, 0x00000000, 80, 4, "PCI_CACHE_LINE", "缓存行大小"),
                RegisterInference(0x10, device_addr, 90, 4, "PCI_BAR0", "基地址寄存器0"),
                RegisterInference(0x14, 0x00000000, 85, 4, "PCI_BAR1", "基地址寄存器1"),
                RegisterInference(0x2C, 0x168c0013, 80, 4, "PCI_SUBSYS_ID", "子系统ID"),
                RegisterInference(0x3C, 0x0000010A, 75, 4, "PCI_INTERRUPT", "中断线"),
            ]
        elif 0xA0000000 <= device_addr <= 0xB0000000:
            # 高地址范围 - 可能是PCIe MMIO空间
            registers = [
                RegisterInference(0x00, 0x168c0013, 90, 4, "DEVICE_ID", "设备标识符"),
                RegisterInference(0x04, 0x00000001, 85, 4, "STATUS_REG", "设备状态寄存器"),
                RegisterInference(0x08, 0x00000000, 80, 4, "CONTROL_REG", "设备控制寄存器"),
                RegisterInference(0x0C, 0x00001000, 75, 4, "BUFFER_SIZE", "缓冲区大小"),
                RegisterInference(0x10, 0x00000000, 70, 4, "TX_DESC", "发送描述符地址"),
                RegisterInference(0x14, 0x00000000, 70, 4, "RX_DESC", "接收描述符地址"),
                RegisterInference(0x18, 0x00000000, 65, 4, "DMA_CTRL", "DMA控制寄存器"),
                RegisterInference(0x1C, 0x0000FFFF, 60, 4, "INT_MASK", "中断屏蔽寄存器"),
            ]
        else:
            # 通用外设
            registers = [
                RegisterInference(0x00, 0x0, 50, 4, "REG_00", "数据寄存器"),
                RegisterInference(0x04, 0x1, 45, 4, "REG_04", "状态寄存器"),
                RegisterInference(0x08, 0x0, 40, 4, "REG_08", "控制寄存器"),
            ]
        
        return InferenceResult(
            device_addr=device_addr,
            register_count=len(registers),
            registers=registers,
            timestamp=int(time.time()),
            inference_id=int(time.time() * 1000) & 0xFFFFFFFF,
            need_resume=True
        )
    
    def perform_ai_inference(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """使用AI进行推断"""
        if not self.ai_inference:
            return None
        
        try:
            # 读取MMIO状态数据
            if hasattr(self.ai_inference, 'read_shared_memory'):
                shared_log = self.ai_inference.read_shared_memory(self.state_mem_name)
                if not shared_log:
                    self.logger.warning("No MMIO state data available for AI inference")
                    return None
            
            device_addr = request.fault_addr & ~0xFFF
            
            # 根据AI系统类型进行推断
            if isinstance(self.ai_inference, LocalPeripheralRegisterInference):
                # 本地AI推断
                results = self.ai_inference.analyze_all_devices(self.state_mem_name)
                if device_addr in results:
                    analysis = results[device_addr]
                    # 解析分析结果生成推断结果
                    return self.parse_ai_analysis(device_addr, analysis)
            else:
                # 云端AI推断
                inferences = self.ai_inference.infer_register_states(self.state_mem_name)
                if device_addr in inferences:
                    analysis = inferences[device_addr]
                    return self.parse_ai_analysis(device_addr, analysis)
            
            return None
            
        except Exception as e:
            self.logger.error(f"AI inference failed: {e}")
            return None
    
    def parse_ai_analysis(self, device_addr: int, analysis: str) -> InferenceResult:
        """解析AI分析结果生成推断结果"""
        # 这里需要解析AI生成的文本，提取寄存器信息
        # 简化版本：生成一些基本的寄存器
        registers = [
            RegisterInference(0x00, 0x0, 70, 4, "AI_REG_00", "AI推断数据寄存器"),
            RegisterInference(0x04, 0x1, 65, 4, "AI_REG_04", "AI推断状态寄存器"),
        ]
        
        return InferenceResult(
            device_addr=device_addr,
            register_count=len(registers),
            registers=registers,
            timestamp=int(time.time()),
            inference_id=int(time.time() * 1000) & 0xFFFFFFFF,
            need_resume=True
        )
    
    def process_inference_request(self, status):
        """处理推断请求"""
        request = InferenceRequest(
            fault_addr=status['fault_addr'],
            fault_pc=status['fault_pc'],
            fault_size=status['fault_size'],
            is_write=status['is_write']
        )
        
        self.logger.info(f"Processing inference request: addr=0x{request.fault_addr:x}, "
                        f"pc=0x{request.fault_pc:x}, size={request.fault_size}, "
                        f"write={request.is_write}")
        
        # 尝试AI推断
        result = self.perform_ai_inference(request)
        
        # 如果AI推断失败，使用规则推断
        if not result:
            self.logger.info("AI inference not available, using rule-based inference")
            result = self.perform_rule_based_inference(request)
        
        # 写入推断结果
        if self.write_inference_result(result):
            self.logger.info(f"Inference completed for device 0x{result.device_addr:x} "
                           f"with {result.register_count} registers")
        else:
            self.logger.error("Failed to write inference result")
    
    def run(self):
        """主运行循环"""
        self.logger.info("Starting inference daemon...")
        
        # 等待桥接共享内存可用
        while self.running and not self.open_bridge_memory():
            self.logger.info("Waiting for bridge memory to be available...")
            time.sleep(1)
        
        if not self.running:
            return
        
        self.logger.info("Inference daemon running, monitoring for MMIO faults...")
        
        while self.running:
            try:
                # 读取桥接状态
                status = self.read_bridge_status()
                if not status:
                    time.sleep(0.1)
                    continue
                
                # 检查是否有待处理的推断请求
                if status['pending_inference'] and status['qemu_paused']:
                    self.process_inference_request(status)
                
                time.sleep(0.1)  # 100ms轮询间隔
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(1)
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("Cleaning up...")
        
        if self.bridge_mem:
            self.bridge_mem.close()
        if self.bridge_fd:
            os.close(self.bridge_fd)
        
        self.logger.info("Inference daemon stopped")

def main():
    parser = argparse.ArgumentParser(description="AI Inference Daemon for QEMU peripheral emulation")
    parser.add_argument("--bridge-mem", default="/mmio_inference_bridge", 
                       help="Bridge shared memory name")
    parser.add_argument("--state-mem", default="/mmio_proxy_shared",
                       help="State shared memory name")
    parser.add_argument("--ai-type", choices=["cloud", "local", "rule"], default="local",
                       help="AI inference type")
    parser.add_argument("--api-key", help="OpenAI API key for cloud inference")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 创建并运行守护进程
    daemon = InferenceDaemon(
        bridge_mem_name=args.bridge_mem,
        state_mem_name=args.state_mem,
        ai_type=args.ai_type,
        api_key=args.api_key
    )
    
    try:
        daemon.run()
    except KeyboardInterrupt:
        print("\nDaemon interrupted by user")
    except Exception as e:
        print(f"Daemon failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
