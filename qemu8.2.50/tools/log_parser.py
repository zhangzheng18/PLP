#!/usr/bin/env python3
"""
PL011 日志解析器 - 将大日志文件拆分成微调训练样本
"""

import re
import json
import os
import hashlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CPUContext:
    pc: str
    cpu_index: int
    cpu_stopped: bool
    cpu_cflags: str
    interrupt_request: str
    in_exclusive_context: bool
    paging_enabled: bool

@dataclass
class AccessContext:
    address: str
    value: str
    size: int
    access_type: str  # READ or WRITE

@dataclass
class RegisterState:
    uartdr: str
    uartrsr: str
    uartfr: str
    uartibrd: str
    uartfbrd: str
    uartlcr_h: str
    uartcr: str
    uartifls: str
    uartimsc: str
    uartris: str
    uartmis: str
    uartdmacr: str

@dataclass
class FIFOState:
    read_count: int
    read_pos: int
    read_trigger: int

@dataclass
class PL011LogEntry:
    operation: str
    timestamp: int
    device_type: str
    sequence_id: int
    cpu_context: CPUContext
    access_context: AccessContext
    register_state: RegisterState
    fifo_state: FIFOState
    
class PL011LogParser:
    def __init__(self):
        self.current_entry = None
        self.parsing_state = "waiting"
        
    def parse_log_file(self, log_file_path: str, output_dir: str, max_samples: int = None):
        """解析日志文件并生成训练样本"""
        
        print(f"开始解析日志文件: {log_file_path}")
        print(f"输出目录: {output_dir}")
        
        # 获取文件大小信息
        file_size = os.path.getsize(log_file_path)
        print(f"文件大小: {file_size / (1024**3):.2f} GB")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/read_samples", exist_ok=True)
        os.makedirs(f"{output_dir}/write_samples", exist_ok=True)
        
        # 统计信息
        total_entries = 0
        read_entries = 0
        write_entries = 0
        failed_entries = 0
        current_entry_lines = []
        processed_bytes = 0
        
        # 分批处理以避免内存问题
        batch_size = 5000  # 每批处理5000个条目
        current_batch = []
        
        print("开始逐行解析...")
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    processed_bytes += len(line.encode('utf-8'))
                    line = line.strip()
                    
                    # 进度显示
                    if line_num % 50000 == 0 and line_num > 0:
                        progress = (processed_bytes / file_size) * 100
                        print(f"已处理 {line_num:,} 行, 进度: {progress:.1f}%, 已解析条目: {total_entries:,}")
                    
                    if line.startswith("=== AI System State Analysis ==="):
                        # 处理之前的条目
                        if current_entry_lines:
                            entry = self._parse_entry(current_entry_lines)
                            if entry:
                                current_batch.append(entry)
                                total_entries += 1
                                if "READ" in entry.access_context.access_type:
                                    read_entries += 1
                                elif "WRITE" in entry.access_context.access_type:
                                    write_entries += 1
                            else:
                                failed_entries += 1
                            
                            # 批量保存以节省内存
                            if len(current_batch) >= batch_size:
                                self._save_batch(current_batch, output_dir)
                                current_batch = []
                            
                            # 检查是否达到最大样本数
                            if max_samples and total_entries >= max_samples:
                                print(f"达到最大样本数限制: {max_samples}")
                                break
                        
                        current_entry_lines = [line]
                    elif current_entry_lines:
                        current_entry_lines.append(line)
                
                # 处理最后一个条目
                if current_entry_lines:
                    entry = self._parse_entry(current_entry_lines)
                    if entry:
                        current_batch.append(entry)
                        total_entries += 1
                        if "READ" in entry.access_context.access_type:
                            read_entries += 1
                        elif "WRITE" in entry.access_context.access_type:
                            write_entries += 1
                    else:
                        failed_entries += 1
                
                # 保存最后一批
                if current_batch:
                    self._save_batch(current_batch, output_dir)
                    
        except Exception as e:
            print(f"文件解析过程中出错: {e}")
            # 保存已解析的数据
            if current_batch:
                self._save_batch(current_batch, output_dir)
        
        # 打印统计信息
        print("=" * 60)
        print(f"解析完成！")
        print(f"总条目数: {total_entries:,}")
        print(f"READ操作: {read_entries:,}")
        print(f"WRITE操作: {write_entries:,}")
        print(f"解析失败: {failed_entries:,}")
        print(f"成功率: {(total_entries/(total_entries+failed_entries)*100):.1f}%")
        print("=" * 60)
        
        return total_entries
    
    def _save_batch(self, entries: List[PL011LogEntry], output_dir: str):
        """批量保存条目以节省内存"""
        read_samples = []
        write_samples = []
        
        for entry in entries:
            try:
                if "READ" in entry.access_context.access_type:
                    sample = self._create_read_sample(entry)
                    read_samples.append(sample)
                elif "WRITE" in entry.access_context.access_type:
                    sample = self._create_write_sample(entry)
                    write_samples.append(sample)
            except Exception as e:
                print(f"创建样本时出错: {e}")
                continue
        
        # 保存读取样本
        if read_samples:
            self._append_samples(read_samples, f"{output_dir}/read_samples")
        
        # 保存写入样本
        if write_samples:
            self._append_samples(write_samples, f"{output_dir}/write_samples")
    
    def _parse_entry(self, lines: List[str]) -> Optional[PL011LogEntry]:
        """解析单个日志条目"""
        try:
            # 检查是否包含必要的标记
            if not any("Operation:" in line for line in lines):
                return None
            
            # 提取基本信息
            operation_line = next((line for line in lines if "Operation:" in line), None)
            if not operation_line:
                return None
                
            operation_match = re.search(r'Operation:\s*(.+)', operation_line)
            operation = operation_match.group(1).strip() if operation_match else ""
            
            # 确保这是PL011相关的操作
            if "PL011" not in operation and "pl011" not in operation:
                return None
            
            # 提取时间戳 - 支持多种格式
            timestamp = 0
            for line in lines:
                if "Timestamp:" in line:
                    timestamp_match = re.search(r'Timestamp:\s*(\d+)', line)
                    if timestamp_match:
                        timestamp = int(timestamp_match.group(1))
                        break
            
            # 提取设备类型
            device_type = "pl011_uart"  # 默认值
            for line in lines:
                if "Device Type:" in line:
                    device_type_match = re.search(r'Device Type:\s*(.+)', line)
                    if device_type_match:
                        device_type = device_type_match.group(1).strip()
                        break
            
            # 提取序列ID
            sequence_id = 0
            for line in lines:
                if "Sequence ID:" in line:
                    sequence_id_match = re.search(r'Sequence ID:\s*(\d+)', line)
                    if sequence_id_match:
                        sequence_id = int(sequence_id_match.group(1))
                        break
            
            # 解析各个组件
            cpu_context = self._parse_cpu_context(lines)
            access_context = self._parse_access_context(lines)
            register_state = self._parse_register_state(lines)
            fifo_state = self._parse_fifo_state(lines)
            
            # 验证关键信息是否存在
            if not access_context.access_type or access_context.access_type not in ["READ", "WRITE"]:
                return None
            
            return PL011LogEntry(
                operation=operation,
                timestamp=timestamp,
                device_type=device_type,
                sequence_id=sequence_id,
                cpu_context=cpu_context,
                access_context=access_context,
                register_state=register_state,
                fifo_state=fifo_state
            )
            
        except Exception as e:
            # 不打印每个错误，避免输出过多
            return None
    
    def _parse_cpu_context(self, lines: List[str]) -> CPUContext:
        """解析CPU上下文"""
        pc = self._extract_field(lines, r'PC: (0x[0-9a-fA-F]+)', "0x0")
        cpu_index = int(self._extract_field(lines, r'CPU Index: (\d+)', "0"))
        cpu_stopped = self._extract_field(lines, r'CPU Stopped: (yes|no)', "no") == "yes"
        cpu_cflags = self._extract_field(lines, r'CPU CFlags: (0x[0-9a-fA-F]+)', "0x0")
        interrupt_request = self._extract_field(lines, r'Interrupt Request: (0x[0-9a-fA-F]+)', "0x0")
        in_exclusive_context = self._extract_field(lines, r'In Exclusive Context: (yes|no)', "no") == "yes"
        paging_enabled = self._extract_field(lines, r'Paging Enabled: (yes|no)', "no") == "yes"
        
        return CPUContext(
            pc=pc,
            cpu_index=cpu_index,
            cpu_stopped=cpu_stopped,
            cpu_cflags=cpu_cflags,
            interrupt_request=interrupt_request,
            in_exclusive_context=in_exclusive_context,
            paging_enabled=paging_enabled
        )
    
    def _parse_access_context(self, lines: List[str]) -> AccessContext:
        """解析访问上下文"""
        # 从Operation行中提取访问类型和偏移
        access_type = "READ"  # 默认值
        address = "0x0"
        value = "0x0"
        size = 4
        
        # 从Operation行提取信息
        for line in lines:
            if "Operation:" in line:
                if "READ" in line.upper():
                    access_type = "READ"
                elif "WRITE" in line.upper():
                    access_type = "WRITE"
                
                # 提取offset
                offset_match = re.search(r'offset[=\s]*(0x[0-9a-fA-F]+)', line, re.IGNORECASE)
                if offset_match:
                    address = offset_match.group(1).lower()
                break
        
        # 从Access行提取详细信息
        address = self._extract_field_flexible(lines, [
            r'Access Address:\s*(0x[0-9a-fA-F]+)',
            r'Address:\s*(0x[0-9a-fA-F]+)',
            r'offset[=\s]*(0x[0-9a-fA-F]+)'
        ], address)
        
        value = self._extract_field_flexible(lines, [
            r'Access Value:\s*(0x[0-9a-fA-F]+)',
            r'Value:\s*(0x[0-9a-fA-F]+)',
            r'value[=\s]*(0x[0-9a-fA-F]+)'
        ], value)
        
        size_str = self._extract_field_flexible(lines, [
            r'Access Size:\s*(\d+)',
            r'Size:\s*(\d+)',
            r'(\d+)\s*bytes?'
        ], str(size))
        
        try:
            size = int(size_str)
        except:
            size = 4
        
        access_type = self._extract_field_flexible(lines, [
            r'Access Type:\s*(READ|WRITE)',
            r'Type:\s*(READ|WRITE)',
            r'\b(READ|WRITE)\b'
        ], access_type)
        
        return AccessContext(
            address=address.lower(),
            value=value.lower(),
            size=size,
            access_type=access_type.upper()
        )
    
    def _parse_register_state(self, lines: List[str]) -> RegisterState:
        """解析寄存器状态"""
        uartdr = self._extract_register_value(lines, "UARTDR")
        uartrsr = self._extract_register_value(lines, "UARTRSR")
        uartfr = self._extract_register_value(lines, "UARTFR")
        uartibrd = self._extract_register_value(lines, "UARTIBRD")
        uartfbrd = self._extract_register_value(lines, "UARTFBRD")
        uartlcr_h = self._extract_register_value(lines, "UARTLCR_H")
        uartcr = self._extract_register_value(lines, "UARTCR")
        uartifls = self._extract_register_value(lines, "UARTIFLS")
        uartimsc = self._extract_register_value(lines, "UARTIMSC")
        uartris = self._extract_register_value(lines, "UARTRIS")
        uartmis = self._extract_register_value(lines, "UARTMIS")
        uartdmacr = self._extract_register_value(lines, "UARTDMACR")
        
        return RegisterState(
            uartdr=uartdr,
            uartrsr=uartrsr,
            uartfr=uartfr,
            uartibrd=uartibrd,
            uartfbrd=uartfbrd,
            uartlcr_h=uartlcr_h,
            uartcr=uartcr,
            uartifls=uartifls,
            uartimsc=uartimsc,
            uartris=uartris,
            uartmis=uartmis,
            uartdmacr=uartdmacr
        )
    
    def _parse_fifo_state(self, lines: List[str]) -> FIFOState:
        """解析FIFO状态"""
        fifo_line = next((line for line in lines if "read_count=" in line), None)
        if fifo_line:
            read_count = int(re.search(r'read_count=(\d+)', fifo_line).group(1))
            read_pos = int(re.search(r'read_pos=(\d+)', fifo_line).group(1))
            read_trigger = int(re.search(r'read_trigger=(\d+)', fifo_line).group(1))
        else:
            read_count = read_pos = read_trigger = 0
            
        return FIFOState(
            read_count=read_count,
            read_pos=read_pos,
            read_trigger=read_trigger
        )
    
    def _extract_field(self, lines: List[str], pattern: str, default: str) -> str:
        """从行中提取字段"""
        for line in lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)
        return default
    
    def _extract_field_flexible(self, lines: List[str], patterns: List[str], default: str) -> str:
        """使用多个模式灵活提取字段"""
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1)
        return default
    
    def _extract_register_value(self, lines: List[str], register_name: str) -> str:
        """提取寄存器值 - 支持多种格式"""
        # 尝试多种可能的模式
        patterns = [
            rf'{register_name}:\s+(0x[0-9a-fA-F]+)',  # 标准格式
            rf'0x[0-9a-fA-F]+\s+{register_name}:\s+(0x[0-9a-fA-F]+)',  # 带地址前缀
            rf'{register_name}\s*=\s*(0x[0-9a-fA-F]+)',  # 等号格式
            rf'{register_name}\s*:\s*(0x[0-9a-fA-F]+)',  # 冒号格式变体
        ]
        
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).lower()  # 统一转换为小写
        
        return "0x00000000"
    

    
    def _create_read_sample(self, entry: PL011LogEntry) -> Dict:
        """创建读取操作的训练样本"""
        # 输入：除了目标寄存器值之外的所有信息
        input_data = {
            "operation_type": "READ",
            "offset": entry.access_context.address,
            "access_size": entry.access_context.size,
            "cpu_context": {
                "pc": entry.cpu_context.pc,
                "cpu_index": entry.cpu_context.cpu_index,
                "cpu_cflags": entry.cpu_context.cpu_cflags,
                "interrupt_request": entry.cpu_context.interrupt_request,
                "in_exclusive_context": entry.cpu_context.in_exclusive_context,
                "paging_enabled": entry.cpu_context.paging_enabled
            },
            "current_registers": {
                "uartdr": entry.register_state.uartdr,
                "uartrsr": entry.register_state.uartrsr,
                "uartfr": entry.register_state.uartfr,
                "uartibrd": entry.register_state.uartibrd,
                "uartfbrd": entry.register_state.uartfbrd,
                "uartlcr_h": entry.register_state.uartlcr_h,
                "uartcr": entry.register_state.uartcr,
                "uartifls": entry.register_state.uartifls,
                "uartimsc": entry.register_state.uartimsc,
                "uartris": entry.register_state.uartris,
                "uartmis": entry.register_state.uartmis,
                "uartdmacr": entry.register_state.uartdmacr
            },
            "fifo_state": {
                "read_count": entry.fifo_state.read_count,
                "read_pos": entry.fifo_state.read_pos,
                "read_trigger": entry.fifo_state.read_trigger
            },
            "timestamp": entry.timestamp,
            "sequence_id": entry.sequence_id
        }
        
        # 输出：读取的值 + 所有寄存器状态
        output_data = {
            "read_value": entry.access_context.value,
            "resulting_registers": {
                "uartdr": entry.register_state.uartdr,
                "uartrsr": entry.register_state.uartrsr,
                "uartfr": entry.register_state.uartfr,
                "uartibrd": entry.register_state.uartibrd,
                "uartfbrd": entry.register_state.uartfbrd,
                "uartlcr_h": entry.register_state.uartlcr_h,
                "uartcr": entry.register_state.uartcr,
                "uartifls": entry.register_state.uartifls,
                "uartimsc": entry.register_state.uartimsc,
                "uartris": entry.register_state.uartris,
                "uartmis": entry.register_state.uartmis,
                "uartdmacr": entry.register_state.uartdmacr
            }
        }
        
        return {
            "input": input_data,
            "output": output_data,
            "metadata": {
                "operation": entry.operation,
                "device_type": entry.device_type
            }
        }
    
    def _create_write_sample(self, entry: PL011LogEntry) -> Dict:
        """创建写入操作的训练样本"""
        # 输入：写入操作的所有上下文信息
        input_data = {
            "operation_type": "WRITE",
            "offset": entry.access_context.address,
            "write_value": entry.access_context.value,
            "access_size": entry.access_context.size,
            "cpu_context": {
                "pc": entry.cpu_context.pc,
                "cpu_index": entry.cpu_context.cpu_index,
                "cpu_cflags": entry.cpu_context.cpu_cflags,
                "interrupt_request": entry.cpu_context.interrupt_request,
                "in_exclusive_context": entry.cpu_context.in_exclusive_context,
                "paging_enabled": entry.cpu_context.paging_enabled
            },
            "previous_registers": {
                "uartdr": entry.register_state.uartdr,
                "uartrsr": entry.register_state.uartrsr,
                "uartfr": entry.register_state.uartfr,
                "uartibrd": entry.register_state.uartibrd,
                "uartfbrd": entry.register_state.uartfbrd,
                "uartlcr_h": entry.register_state.uartlcr_h,
                "uartcr": entry.register_state.uartcr,
                "uartifls": entry.register_state.uartifls,
                "uartimsc": entry.register_state.uartimsc,
                "uartris": entry.register_state.uartris,
                "uartmis": entry.register_state.uartmis,
                "uartdmacr": entry.register_state.uartdmacr
            },
            "fifo_state": {
                "read_count": entry.fifo_state.read_count,
                "read_pos": entry.fifo_state.read_pos,
                "read_trigger": entry.fifo_state.read_trigger
            },
            "timestamp": entry.timestamp,
            "sequence_id": entry.sequence_id
        }
        
        # 输出：写入后的所有寄存器状态
        output_data = {
            "resulting_registers": {
                "uartdr": entry.register_state.uartdr,
                "uartrsr": entry.register_state.uartrsr,
                "uartfr": entry.register_state.uartfr,
                "uartibrd": entry.register_state.uartibrd,
                "uartfbrd": entry.register_state.uartfbrd,
                "uartlcr_h": entry.register_state.uartlcr_h,
                "uartcr": entry.register_state.uartcr,
                "uartifls": entry.register_state.uartifls,
                "uartimsc": entry.register_state.uartimsc,
                "uartris": entry.register_state.uartris,
                "uartmis": entry.register_state.uartmis,
                "uartdmacr": entry.register_state.uartdmacr
            }
        }
        
        return {
            "input": input_data,
            "output": output_data,
            "metadata": {
                "operation": entry.operation,
                "device_type": entry.device_type
            }
        }
    
    def _append_samples(self, samples: List[Dict], output_dir: str):
        """增量保存训练样本"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 每个文件最多保存1000个样本
        samples_per_file = 1000
        
        # 找到现有文件的最大索引
        existing_files = [f for f in os.listdir(output_dir) if f.startswith('samples_') and f.endswith('.jsonl')]
        max_index = -1
        for filename in existing_files:
            try:
                index = int(filename.split('_')[1].split('.')[0])
                max_index = max(max_index, index)
            except:
                continue
        
        # 检查最后一个文件是否已满
        current_file_index = max_index
        current_file_size = 0
        
        if max_index >= 0:
            last_file = f"{output_dir}/samples_{max_index:04d}.jsonl"
            if os.path.exists(last_file):
                with open(last_file, 'r', encoding='utf-8') as f:
                    current_file_size = sum(1 for _ in f)
        
        # 如果最后一个文件已满，创建新文件
        if current_file_size >= samples_per_file:
            current_file_index += 1
            current_file_size = 0
        
        # 保存样本
        for sample in samples:
            if current_file_size >= samples_per_file:
                current_file_index += 1
                current_file_size = 0
            
            output_file = f"{output_dir}/samples_{current_file_index:04d}.jsonl"
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            current_file_size += 1
    
    def _save_samples(self, samples: List[Dict], output_dir: str):
        """保存训练样本到多个文件（保留原方法以向后兼容）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 每个文件最多保存1000个样本，避免单个文件过大
        samples_per_file = 1000
        
        for i in range(0, len(samples), samples_per_file):
            batch = samples[i:i + samples_per_file]
            file_index = i // samples_per_file
            
            output_file = f"{output_dir}/samples_{file_index:04d}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in batch:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"保存了 {len(batch)} 个样本到 {output_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PL011日志解析器')
    parser.add_argument('--input', required=True, help='输入日志文件路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    parser.add_argument('--max-samples', type=int, help='最大样本数量（用于测试）')
    
    args = parser.parse_args()
    
    parser = PL011LogParser()
    num_entries = parser.parse_log_file(args.input, args.output, args.max_samples)
    
    print(f"\n解析完成！总共处理了 {num_entries} 个日志条目")
    print(f"训练样本已保存到: {args.output}")

if __name__ == "__main__":
    main() 
