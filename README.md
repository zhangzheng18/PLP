# QEMU ARM外设AI推断系统

## 项目概述

本项目是一个完整的ARM外设MMIO访问监控和AI推断系统，旨在解决以下核心问题：

**当QEMU ARM虚拟机遇到未定义的外设地址访问时，自动暂停执行，使用AI分析系统状态，推断外设寄存器值，动态创建外设设备，然后恢复执行。**

## 系统架构

### 核心组件

1. **设备发现层** (`mmio_dump.c`)
   - 自动发现QEMU中的所有ARM外设
   - 生成设备映射文件 `device_map.json`
   - 记录设备类型、MMIO地址、IRQ等信息

2. **状态监控层** (`mmio_proxy.c/.h`)
   - MMIO代理设备，拦截外设访问
   - 记录完整系统状态到共享内存
   - 包括CPU寄存器、PC、外设寄存器快照等

3. **中断钩子层** (`gic_arm_hook.c/.h`)
   - 在IRQ触发时dump详细CPU状态
   - 提供IRQ与MMIO访问的关联分析

4. **推断桥接层** (`mmio_inference_bridge.c/.h`) **[新增]**
   - 处理MMIO访问错误
   - 暂停QEMU执行
   - 等待AI推断结果
   - 动态创建推断出的外设
   - 恢复QEMU执行

5. **AI推断层**
   - `ai_register_inference.py`：云端AI推断（OpenAI）
   - `local_ai_inference.py`：本地AI推断（Transformers）
   - `ai_inference_daemon.py`：推断守护进程 **[新增]**

6. **工具层**
   - `shared_mem_reader.c`：共享内存读取工具
   - `example_usage.py`：完整工作流程演示

## 完整工作流程

### 1. 系统初始化

```bash
# 编译QEMU（包含新的推断桥接设备）
cd qemu-8.2.5
./configure --target-list=aarch64-softmmu
make -j$(nproc)

# 编译工具
gcc -o shared_mem_reader shared_mem_reader.c -lrt
```

### 2. 启动推断守护进程

```bash
# 启动AI推断守护进程（后台运行）
python3 ai_inference_daemon.py --ai-type local &

# 或使用云端AI
python3 ai_inference_daemon.py --ai-type cloud --api-key YOUR_OPENAI_KEY &
```

### 3. 启动QEMU与推断桥接

```bash
# 启动包含推断桥接设备的QEMU
./build/qemu-system-aarch64 \
  -M virt -cpu cortex-a57 -m 1G \
  -kernel /path/to/kernel \
  -device inference-device \
  -device mmio-proxy,base=0x9000000,size=0x1000 \
  -nographic
```

### 4. 自动推断过程

当程序访问未定义的外设地址时：

1. **错误捕获**：QEMU捕获MMIO访问错误
2. **暂停执行**：推断桥接设备暂停QEMU
3. **状态收集**：收集当前CPU和系统状态
4. **AI推断**：守护进程分析状态，推断外设结构
5. **设备创建**：动态创建推断出的外设设备
6. **恢复执行**：QEMU继续执行，访问成功

## 关键特性

### 动态外设创建

系统能够根据AI推断结果动态创建外设：

```c
// 推断结果示例
InferenceResult result = {
    .device_addr = 0x9040000,
    .register_count = 3,
    .registers = {
        {0x00, 0x12345678, 90, 4, "DATA_REG", "数据寄存器"},
        {0x04, 0x00000001, 85, 4, "STATUS_REG", "状态寄存器"},
        {0x08, 0x00000000, 80, 4, "CTRL_REG", "控制寄存器"}
    }
};
```

### 多种推断模式

1. **云端AI推断**：使用OpenAI GPT-4进行高精度推断
2. **本地AI推断**：使用本地Transformers模型
3. **规则推断**：基于地址范围和设备类型的规则推断

### 状态共享机制

系统使用共享内存在QEMU和推断进程间通信：

- `mmio_proxy_shared`：MMIO访问状态历史
- `mmio_inference_bridge`：推断请求和结果

## 使用示例

### 基本使用

```bash
# 1. 启动推断守护进程
python3 ai_inference_daemon.py --ai-type local &

# 2. 启动QEMU
./build/qemu-system-aarch64 -M virt -device inference-device -kernel kernel.bin

# 3. 程序运行时访问未定义外设会自动触发推断
```

### 监控推断过程

```bash
# 查看推断日志
tail -f /var/log/inference_daemon.log

# 查看共享内存状态
./shared_mem_reader -m

# 查看推断结果
cat log/ai_inference_results.json
```

### 自定义配置

```bash
# 使用特定的共享内存名称
python3 ai_inference_daemon.py \
  --bridge-mem /custom_inference_bridge \
  --state-mem /custom_proxy_shared

# 设置调试级别
python3 ai_inference_daemon.py --log-level DEBUG
```

## 技术细节

### MMIO错误处理

当程序访问未映射的MMIO地址时：

```c
void inference_handle_mmio_fault(uint64_t addr, uint64_t pc, 
                                uint32_t size, bool is_write) {
    // 填充错误信息
    bridge->fault_addr = addr;
    bridge->fault_pc = pc;
    bridge->pending_inference = 1;
    
    // 暂停QEMU
    qemu_system_debug_request();
}
```

### AI推断接口

守护进程监控共享内存，检测推断请求：

```python
def process_inference_request(self, status):
    request = InferenceRequest(
        fault_addr=status['fault_addr'],
        fault_pc=status['fault_pc'],
        fault_size=status['fault_size'],
        is_write=status['is_write']
    )
    
    # 执行推断
    result = self.perform_ai_inference(request)
    
    # 写回结果
    self.write_inference_result(result)
```

### 动态设备创建

推断结果被应用到系统：

```c
bool create_inferred_device(InferenceDeviceState *s, 
                           const InferenceResult *result) {
    // 创建内存区域
    memory_region_init_io(region, OBJECT(s), &inferred_ops, 
                         device_state, region_name, device_size);
    
    // 添加到系统内存
    memory_region_add_subregion(get_system_memory(), 
                               result->device_addr, region);
    
    return true;
}
```

## 配置说明

### QEMU设备参数

```bash
# 推断桥接设备
-device inference-device,bridge_mem=/custom_bridge

# MMIO代理设备  
-device mmio-proxy,base=0x9000000,size=0x1000,target=/machine/uart0
```

### 共享内存配置

系统使用POSIX共享内存进行通信：

- 状态内存：4KB，存储MMIO访问历史
- 桥接内存：8KB，存储推断请求和结果

## 故障排除

### 常见问题

1. **共享内存权限错误**
   ```bash
   sudo chmod 666 /dev/shm/mmio_*
   ```

2. **AI模型加载失败**
   ```bash
   pip install transformers torch
   ```

3. **QEMU暂停无法恢复**
   ```bash
   # 检查守护进程状态
   ps aux | grep ai_inference_daemon
   ```

### 调试工具

```bash
# 查看共享内存内容
./shared_mem_reader -n /mmio_inference_bridge

# 监控实时状态
./shared_mem_reader -m

# 检查AI推断结果
python3 -c "
import json
with open('log/ai_inference_results.json') as f:
    print(json.dumps(json.load(f), indent=2))
"
```

## 扩展开发

### 添加新的设备类型

在 `ai_inference_daemon.py` 中添加：

```python
def perform_rule_based_inference(self, request):
    if 0xA000000 <= device_addr <= 0xA010000:
        # 新设备类型的推断逻辑
        registers = [...]
```

### 自定义AI模型

实现 `CustomInferenceSystem` 类：

```python
class CustomInferenceSystem:
    def analyze_all_devices(self, shm_name):
        # 自定义推断逻辑
        return results
```

## 性能考虑

- **推断延迟**：典型推断时间 100-500ms
- **内存使用**：每个动态设备约4KB内存
- **CPU开销**：守护进程CPU使用率 < 5%

## 安全考虑

- 共享内存访问权限控制
- 推断结果的置信度检查
- 动态设备的访问边界检查

---

## 贡献

欢迎提交问题和改进建议。请确保：

1. 详细描述问题场景
2. 提供相关日志信息
3. 说明系统配置和版本信息 
