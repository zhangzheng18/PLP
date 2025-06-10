#!/usr/bin/env python3
"""
外设寄存器推断系统使用示例
演示完整的工作流程：设备发现 -> 状态监控 -> AI推断
"""

import os
import sys
import time
import json
import subprocess
import signal
from typing import Dict, List

class PeripheralAnalysisWorkflow:
    def __init__(self):
        self.log_dir = "log"
        self.device_map_file = os.path.join(self.log_dir, "device_map.json")
        self.shared_memory_name = "/mmio_proxy_shared"
        self.qemu_pid = None
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
    def step1_device_discovery(self):
        """步骤1: 设备发现 - 运行QEMU并获取设备映射"""
        print("=== 步骤1: 设备发现阶段 ===")
        print("启动QEMU以获取设备映射信息...")
        
        # 构建QEMU命令（这里需要根据实际情况调整）
        qemu_cmd = [
            "./build/qemu-system-aarch64",
            "-M", "virt",
            "-cpu", "cortex-a57",
            "-m", "1G",
            "-nographic",
            "-kernel", "/path/to/kernel",  # 需要替换为实际路径
            "-append", "console=ttyAMA0",
            "-device", "mmio-proxy,base=0x9000000,size=0x1000,target=/machine/peripheral-anon/pl011@9000000",
            "-no-reboot"
        ]
        
        print(f"QEMU命令: {' '.join(qemu_cmd)}")
        print("等待设备映射生成...")
        
        # 注意：实际使用时需要等待QEMU完全启动并生成设备映射
        # 这里只是示例，实际应该监控设备映射文件的生成
        time.sleep(5)
        
        if os.path.exists(self.device_map_file):
            print(f"✓ 设备映射文件已生成: {self.device_map_file}")
            self.show_device_map()
        else:
            print("✗ 设备映射文件未找到，请检查QEMU启动是否成功")
            return False
        
        return True
    
    def show_device_map(self):
        """显示设备映射信息"""
        try:
            with open(self.device_map_file, 'r') as f:
                device_map = json.load(f)
            
            print("\n发现的设备:")
            for device_key, device_info in device_map.items():
                print(f"  {device_key}:")
                print(f"    类型: {device_info.get('type', 'unknown')}")
                print(f"    路径: {device_info.get('path', 'unknown')}")
                
                if 'mmio_regions' in device_info:
                    for region_key, region in device_info['mmio_regions'].items():
                        base = region.get('base', 'unknown')
                        size = region.get('size', 'unknown')
                        print(f"    MMIO: 0x{base:x} (大小: {size} 字节)")
                
                if 'irq_lines' in device_info:
                    irq_count = len(device_info['irq_lines'])
                    print(f"    IRQ数量: {irq_count}")
                print()
        
        except Exception as e:
            print(f"读取设备映射失败: {e}")
    
    def step2_start_monitoring(self):
        """步骤2: 启动带有MMIO代理的QEMU进行状态监控"""
        print("=== 步骤2: 状态监控阶段 ===")
        print("启动带MMIO代理的QEMU进行状态监控...")
        
        # 读取设备映射以配置代理
        try:
            with open(self.device_map_file, 'r') as f:
                device_map = json.load(f)
        except:
            print("✗ 无法读取设备映射文件")
            return False
        
        # 构建包含MMIO代理的QEMU命令
        qemu_cmd = [
            "./build/qemu-system-aarch64",
            "-M", "virt",
            "-cpu", "cortex-a57", 
            "-m", "1G",
            "-nographic",
            "-kernel", "/path/to/kernel",  # 需要替换
            "-append", "console=ttyAMA0"
        ]
        
        # 为每个发现的设备添加MMIO代理
        for device_key, device_info in device_map.items():
            if 'mmio_regions' in device_info:
                for region_key, region in device_info['mmio_regions'].items():
                    if 'base' in region:
                        base_addr = region['base']
                        size = region.get('size', 0x1000)
                        device_path = device_info.get('path', '')
                        
                        proxy_args = f"mmio-proxy,base=0x{base_addr:x},size=0x{size:x}"
                        if device_path:
                            proxy_args += f",target={device_path}"
                        
                        qemu_cmd.extend(["-device", proxy_args])
        
        print(f"启动监控QEMU: {' '.join(qemu_cmd[:5])}...")
        print("(完整命令包含多个MMIO代理设备)")
        
        # 这里应该实际启动QEMU，但为了示例我们只是模拟
        print("QEMU监控进程已启动（模拟）")
        print("开始收集MMIO访问数据...")
        
        return True
    
    def step3_simulate_activity(self):
        """步骤3: 模拟一些外设活动以生成数据"""
        print("=== 步骤3: 模拟外设活动 ===")
        print("模拟外设访问以生成分析数据...")
        
        # 创建模拟的共享内存数据
        self.create_mock_shared_memory_data()
        
        print("✓ 模拟数据已生成")
        return True
    
    def create_mock_shared_memory_data(self):
        """创建模拟的共享内存数据用于测试"""
        import struct
        
        # 创建模拟的状态日志数据
        mock_data = bytearray(4096)  # SHARED_MEM_SIZE
        
        # 写入头部信息
        entry_count = 10
        write_index = 10
        struct.pack_into('II', mock_data, 0, entry_count, write_index)
        
        # 写入模拟的日志条目
        entry_size = 8 + 4 + 4 + 8 + 8 + 31*8 + 8 + 8 + 4 + 4 + 256
        for i in range(entry_count):
            offset = 8 + i * entry_size
            
            # 模拟UART访问
            timestamp = int(time.time() * 1000000) + i * 1000
            cpu_id = 0
            irq_num = 0xFFFFFFFF  # 无IRQ
            pc = 0x80000000 + i * 4
            sp = 0x80100000
            
            # 模拟寄存器值
            xregs = [i * 0x1000 + j for j in range(31)]
            
            # 模拟MMIO访问
            mmio_addr = 0x9000000 + (i % 4) * 4  # UART寄存器
            mmio_val = 0x12345678 + i
            mmio_size = 4
            is_write = i % 2
            
            # 打包数据
            packed_data = struct.pack('QIIQQ', timestamp, cpu_id, irq_num, pc, sp)
            packed_data += struct.pack('31Q', *xregs)
            packed_data += struct.pack('QQII', mmio_addr, mmio_val, mmio_size, is_write)
            packed_data += b'\x00' * 256  # 外设寄存器数据
            
            mock_data[offset:offset + len(packed_data)] = packed_data
        
        # 写入临时文件（模拟共享内存）
        temp_shm_file = "/tmp/mmio_proxy_shared_mock"
        with open(temp_shm_file, 'wb') as f:
            f.write(mock_data)
        
        print(f"模拟共享内存数据已写入: {temp_shm_file}")
    
    def step4_ai_inference(self):
        """步骤4: 运行AI推断"""
        print("=== 步骤4: AI推断分析 ===")
        
        # 首先尝试本地AI推断
        print("尝试本地AI推断...")
        try:
            from local_ai_inference import LocalPeripheralRegisterInference
            
            inference_system = LocalPeripheralRegisterInference(
                model_name="microsoft/DialoGPT-medium",
                device_map_path=self.device_map_file
            )
            
            # 使用模拟数据进行推断
            results = inference_system.analyze_all_devices("/tmp/mmio_proxy_shared_mock")
            
            if results:
                print("✓ 本地AI推断完成")
                self.display_inference_results(results)
                return True
            else:
                print("本地AI推断未产生结果")
        
        except ImportError:
            print("本地AI模块不可用，跳过本地推断")
        except Exception as e:
            print(f"本地AI推断失败: {e}")
        
        # 如果本地推断不可用，提示云端推断选项
        print("\n如果有OpenAI API密钥，可以使用云端推断:")
        print("python ai_register_inference.py --api-key YOUR_API_KEY --mode analyze")
        
        return False
    
    def display_inference_results(self, results: Dict[int, str]):
        """显示推断结果"""
        print("\n=== AI推断结果 ===")
        for device_addr, analysis in results.items():
            print(f"\n📱 设备 0x{device_addr:x}:")
            print("-" * 80)
            # 只显示前500个字符以节省空间
            analysis_preview = analysis[:500]
            if len(analysis) > 500:
                analysis_preview += "...\n[结果已截断，完整内容请查看日志文件]"
            print(analysis_preview)
            print("-" * 80)
    
    def step5_monitoring_mode(self):
        """步骤5: 实时监控模式"""
        print("=== 步骤5: 实时监控模式 ===")
        print("启动实时监控和推断...")
        
        print("监控命令示例:")
        print("1. 云端AI监控:")
        print("   python ai_register_inference.py --api-key YOUR_KEY --mode monitor")
        print("2. 本地AI监控:")
        print("   python local_ai_inference.py --model microsoft/DialoGPT-medium")
        print("3. 简单状态读取:")
        print("   ./shared_mem_reader -m")
        
        print("\n按 Ctrl+C 停止监控")
        
        try:
            # 模拟监控过程
            for i in range(10):
                print(f"监控周期 {i+1}/10: 检查新的MMIO访问...")
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n监控已停止")
    
    def cleanup(self):
        """清理资源"""
        print("清理临时文件...")
        temp_files = ["/tmp/mmio_proxy_shared_mock"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"已删除: {temp_file}")
    
    def run_complete_workflow(self):
        """运行完整的工作流程"""
        print("外设寄存器推断系统 - 完整工作流程演示")
        print("=" * 60)
        
        try:
            # 步骤1: 设备发现
            if not self.step1_device_discovery():
                print("设备发现失败，创建示例设备映射...")
                self.create_example_device_map()
            
            print("\n" + "="*60)
            
            # 步骤2: 开始监控
            if not self.step2_start_monitoring():
                return
            
            print("\n" + "="*60)
            
            # 步骤3: 模拟活动
            if not self.step3_simulate_activity():
                return
            
            print("\n" + "="*60)
            
            # 步骤4: AI推断
            self.step4_ai_inference()
            
            print("\n" + "="*60)
            
            # 步骤5: 实时监控（可选）
            response = input("\n是否演示实时监控模式？(y/n): ").lower().strip()
            if response == 'y':
                self.step5_monitoring_mode()
            
        finally:
            self.cleanup()
        
        print("\n工作流程演示完成!")
    
    def create_example_device_map(self):
        """创建示例设备映射文件"""
        example_map = {
            "device_0": {
                "type": "pl011",
                "path": "/machine/peripheral-anon/pl011@9000000",
                "compatible": "arm,pl011",
                "mmio_regions": {
                    "mmio_0": {
                        "base": 150994944,  # 0x9000000
                        "size": 4096,
                        "name": "pl011"
                    }
                },
                "irq_lines": {
                    "irq_0": True
                }
            },
            "device_1": {
                "type": "pl061", 
                "path": "/machine/peripheral-anon/pl061@9030000",
                "compatible": "arm,pl061",
                "mmio_regions": {
                    "mmio_0": {
                        "base": 151191552,  # 0x9030000
                        "size": 4096,
                        "name": "pl061"
                    }
                }
            }
        }
        
        with open(self.device_map_file, 'w') as f:
            json.dump(example_map, f, indent=2)
        
        print(f"✓ 示例设备映射已创建: {self.device_map_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="外设寄存器推断系统使用示例")
    parser.add_argument("--step", type=int, choices=[1,2,3,4,5], 
                       help="只运行特定步骤 (1:设备发现, 2:监控, 3:模拟, 4:推断, 5:实时监控)")
    parser.add_argument("--cleanup", action="store_true", help="只执行清理操作")
    
    args = parser.parse_args()
    
    workflow = PeripheralAnalysisWorkflow()
    
    if args.cleanup:
        workflow.cleanup()
        return
    
    if args.step:
        print(f"执行步骤 {args.step}...")
        if args.step == 1:
            workflow.step1_device_discovery()
        elif args.step == 2:
            workflow.step2_start_monitoring()
        elif args.step == 3:
            workflow.step3_simulate_activity()
        elif args.step == 4:
            workflow.step4_ai_inference()
        elif args.step == 5:
            workflow.step5_monitoring_mode()
    else:
        workflow.run_complete_workflow()

if __name__ == "__main__":
    main() 
