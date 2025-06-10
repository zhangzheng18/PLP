#!/usr/bin/env python3
"""
PCI设备推断系统完整示例
演示固件缺少PCI网卡设备的推断和恢复过程
"""

import os
import sys
import time
import subprocess
import signal
import json
from pathlib import Path

class PCIInferenceExample:
    def __init__(self):
        self.qemu_path = "./build/qemu-system-aarch64"
        self.kernel_path = "vmlinux-arm64"  # 需要准备的ARM64内核
        self.daemon_process = None
        self.qemu_process = None
        
    def check_prerequisites(self):
        """检查运行前提条件"""
        print("🔍 检查运行环境...")
        
        # 检查QEMU可执行文件
        if not os.path.exists(self.qemu_path):
            print(f"❌ QEMU不存在: {self.qemu_path}")
            print("请先编译QEMU:")
            print("  ./configure --target-list=aarch64-softmmu")
            print("  make -j$(nproc)")
            return False
        
        # 检查内核文件
        if not os.path.exists(self.kernel_path):
            print(f"❌ 内核文件不存在: {self.kernel_path}")
            print("请准备ARM64内核文件，或使用简单的测试内核")
            return False
        
        # 检查共享内存读取器
        if not os.path.exists("shared_mem_reader"):
            print("❌ 共享内存读取器未编译")
            print("请编译: gcc -o shared_mem_reader shared_mem_reader.c -lrt")
            return False
        
        # 检查Python依赖
        try:
            import mmap
            import struct
        except ImportError as e:
            print(f"❌ Python依赖缺失: {e}")
            return False
        
        print("✅ 环境检查通过")
        return True
    
    def start_inference_daemon(self):
        """启动AI推断守护进程"""
        print("🤖 启动AI推断守护进程...")
        
        cmd = [
            "python3", "ai_inference_daemon.py",
            "--ai-type", "rule",  # 使用规则推断，不需要AI模型
            "--log-level", "INFO",
            "--bridge-mem", "/pci_inference_bridge",
            "--state-mem", "/pci_proxy_shared"
        ]
        
        try:
            self.daemon_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            print(f"✅ 推断守护进程已启动 (PID: {self.daemon_process.pid})")
            
            # 等待守护进程初始化
            time.sleep(2)
            return True
            
        except Exception as e:
            print(f"❌ 启动推断守护进程失败: {e}")
            return False
    
    def start_qemu_with_missing_pci(self):
        """启动QEMU，故意不包含某个PCI设备"""
        print("🖥️  启动QEMU (缺少PCI网卡设备)...")
        
        cmd = [
            self.qemu_path,
            "-M", "virt",
            "-cpu", "cortex-a57",
            "-m", "1G",
            "-kernel", self.kernel_path,
            "-append", "console=ttyAMA0 loglevel=8 ignore_loglevel earlycon",
            "-nographic",
            "-no-reboot",
            
            # 添加推断桥接设备
            "-device", "inference-device,bridge_mem=/pci_inference_bridge",
            
            # 添加MMIO代理监控已知设备
            "-device", "mmio-proxy,base=0x9000000,size=0x1000,shared_mem=/pci_proxy_shared",
            
            # 故意不添加PCIe网卡设备，让内核访问时触发推断
            # 正常情况下会有: -device virtio-net-pci,netdev=net0
            # "-netdev", "user,id=net0"
        ]
        
        print("QEMU命令:")
        print(" ".join(cmd))
        print()
        
        try:
            self.qemu_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            print(f"✅ QEMU已启动 (PID: {self.qemu_process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ 启动QEMU失败: {e}")
            return False
    
    def monitor_inference_process(self):
        """监控推断过程"""
        print("📊 开始监控推断过程...")
        print("等待内核尝试访问缺失的PCI设备...")
        print()
        
        # 监控共享内存状态
        monitor_cmd = ["./shared_mem_reader", "-m", "-n", "/pci_inference_bridge"]
        
        try:
            monitor_process = subprocess.Popen(
                monitor_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            print("共享内存监控器启动成功")
            print("=" * 60)
            
            # 读取监控输出
            start_time = time.time()
            while time.time() - start_time < 60:  # 最多监控60秒
                line = monitor_process.stdout.readline()
                if line:
                    print(f"[监控] {line.strip()}")
                    
                    # 检测到推断完成
                    if "Inference completed" in line:
                        print("🎉 检测到推断完成!")
                        break
                
                time.sleep(0.1)
            
            monitor_process.terminate()
            
        except Exception as e:
            print(f"监控过程出错: {e}")
    
    def demonstrate_pci_access_fault(self):
        """演示PCI访问错误的触发"""
        print("🔧 模拟PCI设备访问...")
        
        # 这里我们创建一个简单的测试程序来访问不存在的PCI地址
        test_script = """
#!/bin/bash
# 模拟内核中PCI设备初始化代码的行为

echo "模拟PCI设备扫描..."

# 模拟访问PCIe配置空间
# 这些地址在没有对应设备时会触发MMIO错误
echo "尝试读取PCI设备配置..."

# 在真实场景中，这些访问会由内核的PCI子系统执行
# 我们这里只是演示概念

echo "PCI Vendor ID访问: 0x10000000"
echo "PCI Command寄存器访问: 0x10000004"  
echo "PCI BAR0访问: 0x10000010"

echo "等待AI推断系统响应..."
sleep 5

echo "检查是否创建了新的PCI设备..."
"""
        
        with open("test_pci_access.sh", "w") as f:
            f.write(test_script)
        
        os.chmod("test_pci_access.sh", 0o755)
        
        print("✅ PCI访问测试脚本已创建")
        print("在真实场景中，内核会自动尝试访问PCI设备")
    
    def check_inference_results(self):
        """检查推断结果"""
        print("📋 检查推断结果...")
        
        # 检查是否生成了推断结果文件
        result_files = [
            "log/ai_inference_results.json",
            "log/local_ai_results.json"
        ]
        
        for result_file in result_files:
            if os.path.exists(result_file):
                print(f"📄 找到推断结果文件: {result_file}")
                try:
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    print("推断结果摘要:")
                    print(f"  设备数量: {results.get('total_devices', 0)}")
                    print(f"  分析时间: {results.get('timestamp', 'unknown')}")
                    
                    # 显示设备详情
                    for device_key, device_data in results.get('devices', {}).items():
                        print(f"  设备 {device_key}:")
                        print(f"    类型: {device_data.get('device_type', 'unknown')}")
                        result_preview = device_data.get('analysis_result', '')[:200]
                        print(f"    分析: {result_preview}...")
                    
                except Exception as e:
                    print(f"读取结果文件失败: {e}")
            else:
                print(f"❌ 推断结果文件不存在: {result_file}")
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理资源...")
        
        # 终止QEMU进程
        if self.qemu_process:
            self.qemu_process.terminate()
            try:
                self.qemu_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.qemu_process.kill()
            print("✅ QEMU进程已终止")
        
        # 终止推断守护进程
        if self.daemon_process:
            self.daemon_process.terminate()
            try:
                self.daemon_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.daemon_process.kill()
            print("✅ 推断守护进程已终止")
        
        # 清理共享内存
        import subprocess
        try:
            subprocess.run(["rm", "-f", "/dev/shm/pci_*"], check=False)
            print("✅ 共享内存已清理")
        except:
            pass
        
        # 清理测试文件
        try:
            os.remove("test_pci_access.sh")
        except:
            pass
    
    def run_complete_example(self):
        """运行完整示例"""
        print("=" * 80)
        print("🚀 PCI设备AI推断系统完整示例")
        print("=" * 80)
        print()
        print("场景描述:")
        print("1. 固件/内核包含PCI网卡驱动代码")
        print("2. 但QEMU中没有配置对应的PCI设备")
        print("3. 内核尝试访问PCI配置空间时触发MMIO错误")
        print("4. AI推断系统分析访问模式，推断设备结构")
        print("5. 动态创建PCI设备，内核继续正常运行")
        print()
        
        try:
            # 检查环境
            if not self.check_prerequisites():
                return False
            
            # 启动推断守护进程
            if not self.start_inference_daemon():
                return False
            
            print("⏱️  等待守护进程完全启动...")
            time.sleep(3)
            
            # 启动QEMU
            if not self.start_qemu_with_missing_pci():
                return False
            
            print("⏱️  等待QEMU启动和内核加载...")
            time.sleep(5)
            
            # 演示PCI访问
            self.demonstrate_pci_access_fault()
            
            # 监控推断过程
            self.monitor_inference_process()
            
            # 检查结果
            self.check_inference_results()
            
            print()
            print("🎊 示例运行完成!")
            print("在真实场景中，内核的PCI驱动会自动触发这个过程")
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️  用户中断执行")
            return False
        except Exception as e:
            print(f"\n❌ 执行过程中出错: {e}")
            return False
        finally:
            self.cleanup()

def main():
    example = PCIInferenceExample()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--setup-only":
        # 仅设置环境，不运行完整示例
        print("仅检查环境设置...")
        example.check_prerequisites()
        return
    
    success = example.run_complete_example()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 
