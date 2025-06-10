#!/usr/bin/env python3
"""
AI-based Peripheral Register State Inference System
利用大模型分析MMIO操作历史，推断外设寄存器状态
"""

import json
import struct
import mmap
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from collections import defaultdict, deque
import openai
from openai import OpenAI

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
    xregs: List[int]  # 31个寄存器
    mmio_addr: int
    mmio_val: int
    mmio_size: int
    is_write: bool
    mmio_regs: bytes  # 256字节外设寄存器快照

@dataclass
class SharedMemoryLog:
    entry_count: int
    write_index: int
    entries: List[StateLogEntry]

@dataclass
class DeviceInfo:
    device_type: str
    path: str
    mmio_regions: Dict[str, Dict]
    irq_lines: Optional[Dict]
    compatible: Optional[str]

class PeripheralRegisterInference:
    def __init__(self, api_key: str, device_map_path: str = "log/device_map.json"):
        """
        初始化推断系统
        :param api_key: OpenAI API密钥
        :param device_map_path: 设备映射文件路径
        """
        self.client = OpenAI(api_key=api_key)
        self.device_map_path = device_map_path
        self.devices: Dict[int, DeviceInfo] = {}
        self.mmio_history: Dict[int, List[StateLogEntry]] = defaultdict(list)
        self.register_patterns: Dict[str, Dict] = {}
        
        # 加载设备映射
        self.load_device_map()
        
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

    def read_shared_memory(self, shm_name: str = "/mmio_proxy_shared") -> Optional[SharedMemoryLog]:
        """读取共享内存中的状态日志"""
        try:
            # 打开共享内存
            shm_fd = os.open(shm_name, os.O_RDONLY)
            shm_data = mmap.mmap(shm_fd, SHARED_MEM_SIZE, access=mmap.ACCESS_READ)
            
            # 解析共享内存结构
            entry_count, write_index = struct.unpack('II', shm_data[:8])
            
            entries = []
            entry_size = 8 + 4 + 4 + 8 + 8 + 31*8 + 8 + 8 + 4 + 4 + 256  # 总结构大小
            
            for i in range(min(entry_count, MAX_LOG_ENTRIES)):
                offset = 8 + i * entry_size
                
                # 解析单个日志条目
                data = shm_data[offset:offset + entry_size]
                (timestamp, cpu_id, irq_num, pc, sp) = struct.unpack('QIIQQ', data[:32])
                
                # 解析寄存器
                xregs = list(struct.unpack('31Q', data[32:280]))
                
                # 解析MMIO信息
                (mmio_addr, mmio_val, mmio_size, is_write) = struct.unpack('QQII', data[280:304])
                
                # 获取外设寄存器快照
                mmio_regs = data[304:560]
                
                entry = StateLogEntry(
                    timestamp=timestamp,
                    cpu_id=cpu_id,
                    irq_num=irq_num,
                    pc=pc,
                    sp=sp,
                    xregs=xregs,
                    mmio_addr=mmio_addr,
                    mmio_val=mmio_val,
                    mmio_size=mmio_size,
                    is_write=bool(is_write),
                    mmio_regs=mmio_regs
                )
                entries.append(entry)
            
            shm_data.close()
            os.close(shm_fd)
            
            return SharedMemoryLog(entry_count, write_index, entries)
            
        except Exception as e:
            print(f"Error reading shared memory: {e}")
            return None

    def analyze_device_access_patterns(self, entries: List[StateLogEntry]) -> Dict[str, any]:
        """分析设备访问模式"""
        patterns = {
            'access_frequency': defaultdict(int),
            'read_write_ratio': defaultdict(lambda: {'reads': 0, 'writes': 0}),
            'register_values': defaultdict(list),
            'access_sequences': [],
            'irq_correlations': defaultdict(list)
        }
        
        for entry in entries:
            addr = entry.mmio_addr
            patterns['access_frequency'][addr] += 1
            
            if entry.is_write:
                patterns['read_write_ratio'][addr]['writes'] += 1
            else:
                patterns['read_write_ratio'][addr]['reads'] += 1
                
            patterns['register_values'][addr].append({
                'value': entry.mmio_val,
                'timestamp': entry.timestamp,
                'pc': entry.pc
            })
            
            patterns['access_sequences'].append({
                'addr': addr,
                'value': entry.mmio_val,
                'is_write': entry.is_write,
                'timestamp': entry.timestamp,
                'irq': entry.irq_num
            })
            
            if entry.irq_num != 0xFFFFFFFF:  # 有效IRQ
                patterns['irq_correlations'][entry.irq_num].append(entry)
        
        return patterns

    def create_inference_prompt(self, device_addr: int, patterns: Dict, recent_entries: List[StateLogEntry]) -> str:
        """为特定设备创建推断提示"""
        device_info = self.devices.get(device_addr, None)
        device_type = device_info.device_type if device_info else "unknown"
        
        prompt = f"""
你是一个ARM嵌入式系统专家，擅长分析外设寄存器状态。请基于以下MMIO访问历史数据，推断设备的当前寄存器状态和行为模式。

设备信息：
- 类型: {device_type}
- 基地址: 0x{device_addr:x}
- 兼容性: {device_info.compatible if device_info else 'unknown'}

访问模式分析：
"""
        
        # 添加访问频率信息
        if patterns['access_frequency']:
            prompt += "\n访问频率统计:\n"
            for addr, freq in sorted(patterns['access_frequency'].items()):
                offset = addr - device_addr if addr >= device_addr else addr
                prompt += f"  偏移 0x{offset:x}: {freq} 次访问\n"
        
        # 添加读写比例
        if patterns['read_write_ratio']:
            prompt += "\n读写操作统计:\n"
            for addr, ratio in patterns['read_write_ratio'].items():
                offset = addr - device_addr if addr >= device_addr else addr
                total = ratio['reads'] + ratio['writes']
                if total > 0:
                    read_pct = ratio['reads'] / total * 100
                    write_pct = ratio['writes'] / total * 100
                    prompt += f"  偏移 0x{offset:x}: 读 {read_pct:.1f}%, 写 {write_pct:.1f}%\n"
        
        # 添加最近的访问序列
        prompt += f"\n最近 {len(recent_entries)} 次访问序列:\n"
        for i, entry in enumerate(recent_entries[-10:]):  # 只显示最后10次
            offset = entry.mmio_addr - device_addr if entry.mmio_addr >= device_addr else entry.mmio_addr
            op = "写入" if entry.is_write else "读取"
            prompt += f"  {i+1}. 偏移 0x{offset:x} {op} 0x{entry.mmio_val:x} (PC: 0x{entry.pc:x})\n"
        
        # 添加IRQ关联信息
        if any(entry.irq_num != 0xFFFFFFFF for entry in recent_entries):
            prompt += "\nIRQ触发关联:\n"
            for entry in recent_entries[-5:]:
                if entry.irq_num != 0xFFFFFFFF:
                    offset = entry.mmio_addr - device_addr if entry.mmio_addr >= device_addr else entry.mmio_addr
                    prompt += f"  IRQ {entry.irq_num} - 偏移 0x{offset:x} = 0x{entry.mmio_val:x}\n"
        
        prompt += """

请基于以上数据，分析并推断：

1. 设备当前的状态（如：空闲、忙碌、错误等）
2. 关键寄存器的可能含义（如：控制寄存器、状态寄存器、数据寄存器等）
3. 寄存器值的变化趋势和模式
4. 可能的设备配置参数
5. 异常或错误状态检测
6. 下一次可能的寄存器访问预测

请用专业但易懂的语言回答，并给出具体的十六进制地址和数值分析。
"""
        
        return prompt

    def get_ai_inference(self, prompt: str) -> str:
        """调用AI模型进行推断"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一个ARM嵌入式系统和外设寄存器专家，擅长分析硬件行为模式和寄存器状态。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI推断失败: {e}"

    def infer_register_states(self, shm_name: str = "/mmio_proxy_shared") -> Dict[int, str]:
        """对所有监控的设备进行寄存器状态推断"""
        print("开始读取共享内存数据...")
        shared_log = self.read_shared_memory(shm_name)
        
        if not shared_log or not shared_log.entries:
            print("没有找到有效的状态日志数据")
            return {}
        
        print(f"读取到 {len(shared_log.entries)} 条状态记录")
        
        # 按设备地址分组访问记录
        device_entries = defaultdict(list)
        for entry in shared_log.entries:
            # 找到对应的设备基地址
            device_base = None
            for base_addr in self.devices.keys():
                if entry.mmio_addr >= base_addr and entry.mmio_addr < base_addr + 0x1000:  # 假设设备大小为4KB
                    device_base = base_addr
                    break
            
            if device_base:
                device_entries[device_base].append(entry)
            else:
                # 创建新的设备条目（基于访问地址的页边界）
                device_base = entry.mmio_addr & ~0xFFF
                device_entries[device_base].append(entry)
        
        print(f"检测到 {len(device_entries)} 个活跃设备")
        
        # 对每个设备进行推断
        inferences = {}
        for device_addr, entries in device_entries.items():
            print(f"\n分析设备 0x{device_addr:x} ({len(entries)} 条记录)...")
            
            # 分析访问模式
            patterns = self.analyze_device_access_patterns(entries)
            
            # 生成推断提示
            prompt = self.create_inference_prompt(device_addr, patterns, entries)
            
            # 进行AI推断
            print(f"正在进行AI推断...")
            inference_result = self.get_ai_inference(prompt)
            inferences[device_addr] = inference_result
            
            print(f"设备 0x{device_addr:x} 推断完成")
        
        return inferences

    def save_inference_results(self, inferences: Dict[int, str], output_path: str = "log/ai_inference_results.json"):
        """保存推断结果"""
        results = {
            "timestamp": int(time.time()),
            "total_devices": len(inferences),
            "devices": {}
        }
        
        for device_addr, inference in inferences.items():
            device_info = self.devices.get(device_addr, None)
            results["devices"][f"0x{device_addr:x}"] = {
                "device_type": device_info.device_type if device_info else "unknown",
                "inference_result": inference,
                "analysis_time": int(time.time())
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"推断结果已保存到 {output_path}")

    def real_time_monitoring(self, shm_name: str = "/mmio_proxy_shared", interval: int = 5):
        """实时监控和推断"""
        print(f"开始实时监控 (每 {interval} 秒分析一次)")
        last_entry_count = 0
        
        while True:
            try:
                shared_log = self.read_shared_memory(shm_name)
                if shared_log and shared_log.entry_count > last_entry_count:
                    print(f"\n检测到新的访问记录 ({shared_log.entry_count - last_entry_count} 条新记录)")
                    
                    # 只分析新的记录
                    new_entries = shared_log.entries[last_entry_count:]
                    if new_entries:
                        # 快速分析最近的活动
                        recent_addrs = set(entry.mmio_addr for entry in new_entries[-10:])
                        print(f"最近访问的地址: {[hex(addr) for addr in recent_addrs]}")
                        
                        # 检测异常模式
                        error_patterns = self.detect_error_patterns(new_entries)
                        if error_patterns:
                            print(f"⚠️  检测到异常模式: {error_patterns}")
                    
                    last_entry_count = shared_log.entry_count
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n监控已停止")
                break
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(interval)

    def detect_error_patterns(self, entries: List[StateLogEntry]) -> List[str]:
        """检测错误模式"""
        patterns = []
        
        # 检测重复失败的写操作
        write_failures = defaultdict(int)
        for entry in entries:
            if entry.is_write and entry.mmio_val == 0:  # 假设写0可能表示错误
                write_failures[entry.mmio_addr] += 1
        
        for addr, count in write_failures.items():
            if count > 3:
                patterns.append(f"地址 0x{addr:x} 连续写入失败 {count} 次")
        
        # 检测异常高频访问
        addr_counts = defaultdict(int)
        for entry in entries[-20:]:  # 检查最近20次访问
            addr_counts[entry.mmio_addr] += 1
        
        for addr, count in addr_counts.items():
            if count > 10:
                patterns.append(f"地址 0x{addr:x} 异常高频访问 ({count} 次)")
        
        return patterns

def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-based Peripheral Register Inference")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--device-map", default="log/device_map.json", help="Device map file path")
    parser.add_argument("--shm-name", default="/mmio_proxy_shared", help="Shared memory name")
    parser.add_argument("--mode", choices=["analyze", "monitor"], default="analyze", help="Operation mode")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    # 创建推断系统
    inference_system = PeripheralRegisterInference(args.api_key, args.device_map)
    
    if args.mode == "analyze":
        # 一次性分析模式
        print("=== AI外设寄存器推断系统 ===")
        inferences = inference_system.infer_register_states(args.shm_name)
        
        if inferences:
            print("\n=== 推断结果 ===")
            for device_addr, result in inferences.items():
                print(f"\n📱 设备 0x{device_addr:x}:")
                print("-" * 60)
                print(result)
                print("-" * 60)
            
            # 保存结果
            inference_system.save_inference_results(inferences)
        else:
            print("没有可分析的数据")
    
    elif args.mode == "monitor":
        # 实时监控模式
        inference_system.real_time_monitoring(args.shm_name, args.interval)

if __name__ == "__main__":
    main() 
