#!/usr/bin/env python3
"""
本地AI推断系统 - 使用本地大模型进行外设寄存器状态推断
不依赖云端API，可在离线环境中使用
"""

import json
import struct
import mmap
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    TextGenerationPipeline
)

# 共享内存结构定义（与C代码保持一致）
SHARED_MEM_SIZE = 4096
MAX_LOG_ENTRIES = 100

@dataclass
class StateLogEntry:
    timestamp: int
    cpu_id: int
    irq_num: int
    pc: int
    sp: int
    xregs: List[int]
    mmio_addr: int
    mmio_val: int
    mmio_size: int
    is_write: bool
    mmio_regs: bytes

@dataclass
class DeviceInfo:
    device_type: str
    path: str
    mmio_regions: Dict[str, Dict]
    irq_lines: Optional[Dict]
    compatible: Optional[str]

class LocalPeripheralRegisterInference:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device_map_path: str = "log/device_map.json"):
        """
        初始化本地推断系统
        :param model_name: 本地模型名称（可以是Hugging Face模型或本地路径）
        :param device_map_path: 设备映射文件路径
        """
        self.device_map_path = device_map_path
        self.devices: Dict[int, DeviceInfo] = {}
        
        # 初始化本地模型
        print(f"正在加载本地模型: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # 创建文本生成管道
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用规则基推断系统")
            self.generator = None
        
        # 加载设备映射
        self.load_device_map()
        
        # 初始化已知设备模式库
        self.init_device_patterns()
        
    def init_device_patterns(self):
        """初始化已知设备的寄存器模式"""
        self.device_patterns = {
            'pl011': {
                'registers': {
                    0x00: 'UARTDR - 数据寄存器',
                    0x04: 'UARTRSR/UARTECR - 状态/错误清除寄存器',
                    0x18: 'UARTFR - 标志寄存器',
                    0x20: 'UARTILPR - IrDA低功耗计数器寄存器',
                    0x24: 'UARTIBRD - 整数波特率分频寄存器',
                    0x28: 'UARTFBRD - 小数波特率分频寄存器',
                    0x2C: 'UARTLCR_H - 线控制寄存器',
                    0x30: 'UARTCR - 控制寄存器',
                    0x34: 'UARTIFLS - 中断FIFO级别选择寄存器',
                    0x38: 'UARTIMSC - 中断屏蔽设置/清除寄存器',
                    0x3C: 'UARTRIS - 原始中断状态寄存器',
                    0x40: 'UARTMIS - 屏蔽中断状态寄存器',
                    0x44: 'UARTICR - 中断清除寄存器'
                },
                'patterns': {
                    'tx_ready': {'reg': 0x18, 'mask': 0x20, 'value': 0x20},
                    'rx_ready': {'reg': 0x18, 'mask': 0x10, 'value': 0x00},
                    'busy': {'reg': 0x18, 'mask': 0x08, 'value': 0x08}
                }
            },
            'pl061': {
                'registers': {
                    0x000: 'GPIODATA - 数据寄存器',
                    0x400: 'GPIODIR - 方向寄存器',
                    0x404: 'GPIOIS - 中断感知寄存器',
                    0x408: 'GPIOIBE - 中断双边沿寄存器',
                    0x40C: 'GPIOIEV - 中断事件寄存器',
                    0x410: 'GPIOIE - 中断屏蔽寄存器',
                    0x414: 'GPIORIS - 原始中断状态寄存器',
                    0x418: 'GPIOMIS - 屏蔽中断状态寄存器',
                    0x41C: 'GPIOICR - 中断清除寄存器'
                }
            }
        }

    def load_device_map(self):
        """加载设备映射信息"""
        if not os.path.exists(self.device_map_path):
            print(f"Warning: Device map file {self.device_map_path} not found")
            return
            
        with open(self.device_map_path, 'r') as f:
            device_data = json.load(f)
            
        for key, device in device_data.items():
            if 'mmio_regions' in device:
                for region_key, region in device['mmio_regions'].items():
                    if 'base' in region:
                        base_addr = region['base']
                        self.devices[base_addr] = DeviceInfo(
                            device_type=device.get('type', 'unknown'),
                            path=device.get('path', ''),
                            mmio_regions=device.get('mmio_regions', {}),
                            irq_lines=device.get('irq_lines'),
                            compatible=device.get('compatible')
                        )
        
        print(f"Loaded {len(self.devices)} devices from device map")

    def read_shared_memory(self, shm_name: str = "/mmio_proxy_shared") -> Optional[List[StateLogEntry]]:
        """读取共享内存中的状态日志"""
        try:
            shm_fd = os.open(shm_name, os.O_RDONLY)
            shm_data = mmap.mmap(shm_fd, SHARED_MEM_SIZE, access=mmap.ACCESS_READ)
            
            entry_count, write_index = struct.unpack('II', shm_data[:8])
            
            entries = []
            entry_size = 8 + 4 + 4 + 8 + 8 + 31*8 + 8 + 8 + 4 + 4 + 256
            
            for i in range(min(entry_count, MAX_LOG_ENTRIES)):
                offset = 8 + i * entry_size
                data = shm_data[offset:offset + entry_size]
                
                (timestamp, cpu_id, irq_num, pc, sp) = struct.unpack('QIIQQ', data[:32])
                xregs = list(struct.unpack('31Q', data[32:280]))
                (mmio_addr, mmio_val, mmio_size, is_write) = struct.unpack('QQII', data[280:304])
                mmio_regs = data[304:560]
                
                entry = StateLogEntry(
                    timestamp=timestamp, cpu_id=cpu_id, irq_num=irq_num,
                    pc=pc, sp=sp, xregs=xregs, mmio_addr=mmio_addr,
                    mmio_val=mmio_val, mmio_size=mmio_size,
                    is_write=bool(is_write), mmio_regs=mmio_regs
                )
                entries.append(entry)
            
            shm_data.close()
            os.close(shm_fd)
            return entries
            
        except Exception as e:
            print(f"Error reading shared memory: {e}")
            return None

    def analyze_device_with_rules(self, device_addr: int, entries: List[StateLogEntry]) -> str:
        """基于规则的设备分析"""
        device_info = self.devices.get(device_addr, None)
        device_type = device_info.device_type if device_info else "unknown"
        
        analysis = f"=== 设备分析报告 ===\n"
        analysis += f"设备类型: {device_type}\n"
        analysis += f"基地址: 0x{device_addr:x}\n"
        analysis += f"分析的访问记录: {len(entries)} 条\n\n"
        
        # 统计访问模式
        read_count = sum(1 for e in entries if not e.is_write)
        write_count = sum(1 for e in entries if e.is_write)
        
        analysis += f"读写统计:\n"
        analysis += f"  读操作: {read_count} 次\n"
        analysis += f"  写操作: {write_count} 次\n"
        analysis += f"  读写比: {read_count/(write_count+1):.2f}\n\n"
        
        # 分析访问的寄存器
        register_access = defaultdict(list)
        for entry in entries:
            offset = entry.mmio_addr - device_addr
            register_access[offset].append(entry)
        
        analysis += f"寄存器访问分析:\n"
        for offset in sorted(register_access.keys()):
            entries_for_reg = register_access[offset]
            values = [e.mmio_val for e in entries_for_reg]
            unique_values = set(values)
            
            analysis += f"  偏移 0x{offset:x}:\n"
            analysis += f"    访问次数: {len(entries_for_reg)}\n"
            analysis += f"    唯一值数量: {len(unique_values)}\n"
            analysis += f"    值范围: 0x{min(values):x} - 0x{max(values):x}\n"
            
            # 查找已知寄存器定义
            device_pattern = None
            for pattern_name, pattern_data in self.device_patterns.items():
                if pattern_name in device_type.lower():
                    device_pattern = pattern_data
                    break
            
            if device_pattern and offset in device_pattern['registers']:
                analysis += f"    寄存器名称: {device_pattern['registers'][offset]}\n"
            
            # 分析值的模式
            if len(unique_values) == 1:
                analysis += f"    模式: 固定值 0x{list(unique_values)[0]:x}\n"
            elif len(unique_values) <= 5:
                analysis += f"    模式: 有限状态值 {[hex(v) for v in sorted(unique_values)]}\n"
            else:
                analysis += f"    模式: 动态变化值\n"
        
        # 检测设备状态
        analysis += f"\n设备状态推断:\n"
        
        # 特定设备类型的分析
        if 'pl011' in device_type.lower():
            analysis += self.analyze_uart_state(register_access, device_addr)
        elif 'pl061' in device_type.lower():
            analysis += self.analyze_gpio_state(register_access, device_addr)
        else:
            analysis += "  通用外设，无特定状态分析\n"
        
        # IRQ分析
        irq_entries = [e for e in entries if e.irq_num != 0xFFFFFFFF]
        if irq_entries:
            analysis += f"\nIRQ活动分析:\n"
            irq_counts = defaultdict(int)
            for entry in irq_entries:
                irq_counts[entry.irq_num] += 1
            
            for irq_num, count in irq_counts.items():
                analysis += f"  IRQ {irq_num}: 触发 {count} 次\n"
        
        return analysis

    def analyze_uart_state(self, register_access: Dict[int, List], device_addr: int) -> str:
        """UART设备特定分析"""
        analysis = "  UART设备状态分析:\n"
        
        # 分析控制寄存器 (0x30)
        if 0x30 in register_access:
            control_values = [e.mmio_val for e in register_access[0x30]]
            latest_control = control_values[-1] if control_values else 0
            
            analysis += f"    控制寄存器(UARTCR): 0x{latest_control:x}\n"
            if latest_control & 0x01:
                analysis += "      - UART已启用\n"
            if latest_control & 0x100:
                analysis += "      - 发送使能\n"
            if latest_control & 0x200:
                analysis += "      - 接收使能\n"
        
        # 分析标志寄存器 (0x18)
        if 0x18 in register_access:
            flag_values = [e.mmio_val for e in register_access[0x18]]
            latest_flags = flag_values[-1] if flag_values else 0
            
            analysis += f"    标志寄存器(UARTFR): 0x{latest_flags:x}\n"
            if latest_flags & 0x08:
                analysis += "      - UART忙碌\n"
            if latest_flags & 0x10:
                analysis += "      - 接收FIFO为空\n"
            if latest_flags & 0x20:
                analysis += "      - 发送FIFO满\n"
            if latest_flags & 0x80:
                analysis += "      - 发送FIFO为空\n"
        
        # 分析数据传输
        if 0x00 in register_access:
            data_accesses = register_access[0x00]
            writes = [e for e in data_accesses if e.is_write]
            reads = [e for e in data_accesses if not e.is_write]
            
            analysis += f"    数据传输统计:\n"
            analysis += f"      - 发送字节数: {len(writes)}\n"
            analysis += f"      - 接收字节数: {len(reads)}\n"
            
            if writes:
                sent_data = [e.mmio_val & 0xFF for e in writes]
                analysis += f"      - 最近发送的数据: {[hex(d) for d in sent_data[-5:]]}\n"
        
        return analysis

    def analyze_gpio_state(self, register_access: Dict[int, List], device_addr: int) -> str:
        """GPIO设备特定分析"""
        analysis = "  GPIO设备状态分析:\n"
        
        # 分析方向寄存器 (0x400)
        if 0x400 in register_access:
            dir_values = [e.mmio_val for e in register_access[0x400]]
            latest_dir = dir_values[-1] if dir_values else 0
            
            analysis += f"    方向寄存器(GPIODIR): 0x{latest_dir:x}\n"
            for i in range(8):
                if latest_dir & (1 << i):
                    analysis += f"      - GPIO{i}: 输出\n"
                else:
                    analysis += f"      - GPIO{i}: 输入\n"
        
        # 分析数据寄存器 (0x000)
        if 0x000 in register_access:
            data_values = [e.mmio_val for e in register_access[0x000]]
            if data_values:
                latest_data = data_values[-1]
                analysis += f"    数据寄存器(GPIODATA): 0x{latest_data:x}\n"
                for i in range(8):
                    if latest_data & (1 << i):
                        analysis += f"      - GPIO{i}: 高电平\n"
                    else:
                        analysis += f"      - GPIO{i}: 低电平\n"
        
        return analysis

    def generate_ai_inference(self, prompt: str) -> str:
        """使用本地AI模型生成推断"""
        if not self.generator:
            return "本地AI模型不可用，使用规则基分析"
        
        try:
            # 构建适合模型的提示
            formatted_prompt = f"分析以下嵌入式系统外设数据:\n{prompt}\n\n分析结果:"
            
            # 生成回复
            response = self.generator(
                formatted_prompt,
                max_length=len(formatted_prompt) + 500,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            # 提取生成的部分（去掉原始提示）
            result = generated_text[len(formatted_prompt):].strip()
            
            return result if result else "AI模型未能生成有效分析"
            
        except Exception as e:
            return f"AI推断过程出错: {e}"

    def analyze_all_devices(self, shm_name: str = "/mmio_proxy_shared") -> Dict[int, str]:
        """分析所有设备"""
        print("开始读取共享内存数据...")
        entries = self.read_shared_memory(shm_name)
        
        if not entries:
            print("没有找到有效的状态日志数据")
            return {}
        
        print(f"读取到 {len(entries)} 条状态记录")
        
        # 按设备地址分组
        device_entries = defaultdict(list)
        for entry in entries:
            device_base = None
            for base_addr in self.devices.keys():
                if entry.mmio_addr >= base_addr and entry.mmio_addr < base_addr + 0x1000:
                    device_base = base_addr
                    break
            
            if device_base:
                device_entries[device_base].append(entry)
            else:
                device_base = entry.mmio_addr & ~0xFFF
                device_entries[device_base].append(entry)
        
        print(f"检测到 {len(device_entries)} 个活跃设备")
        
        # 分析每个设备
        results = {}
        for device_addr, device_entry_list in device_entries.items():
            print(f"\n分析设备 0x{device_addr:x} ({len(device_entry_list)} 条记录)...")
            
            # 使用规则基分析
            rule_analysis = self.analyze_device_with_rules(device_addr, device_entry_list)
            
            # 如果有AI模型，也尝试AI分析
            if self.generator:
                print("正在进行AI增强分析...")
                ai_analysis = self.generate_ai_inference(rule_analysis)
                combined_analysis = f"{rule_analysis}\n\n=== AI增强分析 ===\n{ai_analysis}"
                results[device_addr] = combined_analysis
            else:
                results[device_addr] = rule_analysis
            
            print(f"设备 0x{device_addr:x} 分析完成")
        
        return results

    def save_results(self, results: Dict[int, str], output_path: str = "log/local_ai_results.json"):
        """保存分析结果"""
        output_data = {
            "timestamp": int(time.time()),
            "analysis_type": "local_ai_inference",
            "total_devices": len(results),
            "devices": {}
        }
        
        for device_addr, analysis in results.items():
            device_info = self.devices.get(device_addr, None)
            output_data["devices"][f"0x{device_addr:x}"] = {
                "device_type": device_info.device_type if device_info else "unknown",
                "analysis_result": analysis,
                "timestamp": int(time.time())
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到 {output_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local AI-based Peripheral Register Inference")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="本地模型名称或路径")
    parser.add_argument("--device-map", default="log/device_map.json", help="设备映射文件路径")
    parser.add_argument("--shm-name", default="/mmio_proxy_shared", help="共享内存名称")
    parser.add_argument("--output", default="log/local_ai_results.json", help="输出文件路径")
    
    args = parser.parse_args()
    
    print("=== 本地AI外设寄存器推断系统 ===")
    
    # 创建推断系统
    inference_system = LocalPeripheralRegisterInference(args.model, args.device_map)
    
    # 执行分析
    results = inference_system.analyze_all_devices(args.shm_name)
    
    if results:
        print("\n=== 分析结果 ===")
        for device_addr, analysis in results.items():
            print(f"\n📱 设备 0x{device_addr:x}:")
            print("=" * 80)
            print(analysis)
            print("=" * 80)
        
        # 保存结果
        inference_system.save_results(results, args.output)
    else:
        print("没有可分析的数据")

if __name__ == "__main__":
    main() 
