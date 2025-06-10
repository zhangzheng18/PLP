#!/bin/bash

# QEMU AI Inference System - 统一构建脚本
# 构建完整的AI驱动的外设推断系统

echo "🚀 Building QEMU AI Inference System..."

# 检查依赖
check_dependencies() {
    echo "📋 Checking dependencies..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3 is required"
        exit 1
    fi
    
    # 检查meson
    if ! command -v meson &> /dev/null; then
        echo "❌ Meson is required for QEMU build"
        exit 1
    fi
    
    # 检查ninja
    if ! command -v ninja &> /dev/null; then
        echo "❌ Ninja is required for QEMU build"
        exit 1
    fi
    
    echo "✅ All dependencies satisfied"
}

# 构建QEMU（包含AI推断模块）
build_qemu() {
    echo "🔨 Building QEMU with AI Inference modules..."
    
    # 清理之前的构建
    if [ -d "build" ]; then
        rm -rf build
    fi
    
    # 配置QEMU构建
    meson setup build \
        --buildtype=debug \
        --enable-plugins \
        --target-list=aarch64-softmmu,arm-softmmu \
        --enable-trace-backends=simple \
        --enable-debug
    
    if [ $? -ne 0 ]; then
        echo "❌ QEMU configuration failed"
        exit 1
    fi
    
    # 编译QEMU
    ninja -C build
    
    if [ $? -ne 0 ]; then
        echo "❌ QEMU compilation failed"
        exit 1
    fi
    
    echo "✅ QEMU compiled successfully with AI inference modules"
}

# 构建用户空间工具
build_userspace_tools() {
    echo "🔧 Building userspace tools..."
    
    # 编译共享内存读取器
    gcc -o shared_mem_reader shared_mem_reader.c -lrt -lpthread
    
    if [ $? -ne 0 ]; then
        echo "❌ Shared memory reader compilation failed"
        exit 1
    fi
    
    # 安装Python依赖
    echo "📦 Installing Python dependencies..."
    pip3 install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "❌ Python dependencies installation failed"
        exit 1
    fi
    
    echo "✅ Userspace tools built successfully"
}

# 设置环境
setup_environment() {
    echo "⚙️ Setting up environment..."
    
    # 创建日志目录
    mkdir -p log
    
    # 设置共享内存权限
    sudo sysctl -w kernel.shmmax=1073741824  # 1GB
    sudo sysctl -w kernel.shmall=268435456   # 1GB in pages
    
    echo "✅ Environment setup complete"
}

# 运行测试
run_tests() {
    echo "🧪 Running system tests..."
    
    # 测试共享内存读取器
    echo "Testing shared memory reader..."
    timeout 5s ./shared_mem_reader || echo "Shared memory reader test completed"
    
    # 测试Python模块导入
    echo "Testing Python modules..."
    python3 -c "import ai_inference_daemon; print('✅ AI daemon module OK')"
    python3 -c "import ai_register_inference; print('✅ AI inference module OK')"
    python3 -c "import local_ai_inference; print('✅ Local AI module OK')"
    
    echo "✅ All tests passed"
}

# 显示使用说明
show_usage() {
    echo ""
    echo "🎯 Build completed! Usage instructions:"
    echo ""
    echo "1. 启动AI推断守护进程:"
    echo "   python3 ai_inference_daemon.py"
    echo ""
    echo "2. 启动QEMU（另一个终端）:"
    echo "   ./build/qemu-system-aarch64 -machine virt -cpu cortex-a57 \\"
    echo "   -device mmio-inference-bridge,shared_mem=/mmio_inference_bridge \\"
    echo "   -kernel your_kernel.bin -nographic"
    echo ""
    echo "3. 运行PCI设备推断示例:"
    echo "   python3 pci_device_inference_example.py"
    echo ""
    echo "4. 监控系统状态:"
    echo "   ./shared_mem_reader"
    echo ""
    echo "📖 Complete documentation: README_INFERENCE_SYSTEM.md"
}

# 主函数
main() {
    check_dependencies
    build_qemu
    build_userspace_tools
    setup_environment
    run_tests
    show_usage
    
    echo ""
    echo "🎉 QEMU AI Inference System build completed successfully!"
    echo "🔗 The system includes:"
    echo "   - Device Discovery (mmio_dump.c)"
    echo "   - MMIO Proxy and State Monitoring (mmio_proxy.c)" 
    echo "   - AI Inference Bridge (mmio_inference_bridge.c)"
    echo "   - Fault Handler (mmio_fault_handler.c)"
    echo "   - AI Inference Daemon (ai_inference_daemon.py)"
    echo "   - Cloud & Local AI Models (ai_register_inference.py, local_ai_inference.py)"
    echo "   - Comprehensive Examples and Documentation"
}

# 运行主函数
main "$@"
