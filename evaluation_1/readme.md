# JSON数据处理器更新摘要

## 概述

本文档详细说明了为适配重构后的 `ai_inference_engine.py` 而对 `json_data_processor.py` 进行的全面更新。

## 主要更新内容

### 1. 导入模块更新

**原代码：**
```python
from ai_inference_engine import UniversalinferenceEngine, InferenceConfig
```

**新代码：**
```python
from ai_inference_engine import (
    MultiLevelInferenceEngine, 
    InferenceConfig, 
    InferenceConstraints,
    AccessPattern,
    DeviceConfig,
    PeripheralType
)
```

**更新说明：**
- 主推理引擎从 `UniversalInferenceEngine` 改为 `MultiLevelInferenceEngine`
- 新增导入 `InferenceConstraints` 用于约束配置
- 新增导入 `AccessPattern` 用于访问模式定义
- 新增导入 `DeviceConfig` 和 `PeripheralType` 支持设备配置

### 2. 配置结构重构

**原代码：**
```python
self.ai_config = InferenceConfig(
    timeout_seconds=120,
    enable_validation=False
)
```

**新代码：**
```python
constraints = InferenceConstraints(
    min_registers_count=3,
    max_registers_count=15,
    required_registers={"UARTDR", "UARTFR", "UARTCR"},
    register_value_range=(0x0, 0xFFFFFFFF)
)

self.ai_config = InferenceConfig(
    model_path="/data/LLM_models/Qwen2.5-Coder-14B-Instruct",
    temperature=0.2,
    timeout_seconds=120,
    confidence_threshold=0.6,
    constraints=constraints,
    enable_pattern_cache=True,
    enable_knowledge_base=True,
    enable_semantic_analysis=True
)
```

**更新说明：**
- 增加了推理约束配置，包括寄存器数量限制、必需寄存器、值范围等
- 新增模型路径、温度、置信度阈值等AI参数
- 启用了模式缓存、知识库和语义分析功能

### 3. 数据提取器增强

#### 3.1 新增访问模式创建方法

```python
def create_access_pattern(self, sample: Dict[str, Any]) -> AccessPattern:
    """创建访问模式对象供新推理引擎使用"""
    metadata = sample["metadata"]
    
    # 构建访问序列
    sequence_item = {
        "operation": sample["operation"],
        "register": self._extract_register_name(metadata.get("offset", "unknown")),
        "type": metadata.get("operation_type", "UNKNOWN"),
        "timestamp": metadata.get("timestamp", time.time()),
        "value": metadata.get("value"),
        "current_registers": sample["current_registers"]
    }
    
    # 创建访问模式
    access_pattern = AccessPattern(
        sequence=[sequence_item],
        timestamp=metadata.get("timestamp", time.time()),
        context={
            "device_type": metadata.get("device_type", "pl011_uart"),
            "operation": sample["operation"],
            "current_registers": sample["current_registers"]
        }
    )
    
    return access_pattern
```

#### 3.2 新增寄存器名称映射

```python
def _extract_register_name(self, offset: str) -> str:
    """根据偏移量提取寄存器名称"""
    offset_map = {
        "0x00": "UARTDR",
        "0x04": "UARTRSR", 
        "0x18": "UARTFR",
        "0x24": "UARTIBRD",
        "0x28": "UARTFBRD", 
        "0x2C": "UARTLCR_H",
        "0x30": "UARTCR",
        "0x34": "UARTIFLS",
        "0x38": "UARTIMSC",
        "0x3C": "UARTRIS",
        "0x40": "UARTMIS",
        "0x48": "UARTDMACR"
    }
    
    return offset_map.get(offset, "UNKNOWN_REG")
```

### 4. 推理引擎调用方式更新

**原代码：**
```python
inference_input = {
    "operation": sample["operation"],
    "current_registers": sample["current_registers"]
}

ai_result = self.ai_engine.predict(inference_input)
```

**新代码：**
```python
access_pattern = self.extractor.create_access_pattern(sample)

ai_result = self.ai_engine.intelligent_inference(
    access_pattern=access_pattern,
    device_name="PL011"
)
```

**更新说明：**
- 调用方法从 `predict()` 改为 `intelligent_inference()`
- 输入从简单字典改为结构化的 `AccessPattern` 对象
- 增加了设备名称参数

### 5. 返回值结构处理更新

**原代码：**
```python
prediction = ai_result["prediction"]
```

**新代码：**
```python
prediction = {
    "read_value": ai_result.get("read_value"),
    "registers": ai_result.get("registers", {})
}
```

**更新说明：**
- 新引擎直接返回 `read_value` 和 `registers` 字段，无需嵌套在 `prediction` 中
- 适配了新的标准化返回格式

### 6. 结果统计增强

#### 6.1 新增置信度和引擎特性统计

```python
confidences = [r.get("confidence", 0.0) for r in successful_results]

# 统计快速路径使用情况
quick_path_count = len([r for r in successful_results if r.get("quick_path", False)])
auto_fixed_count = len([r for r in successful_results if r.get("auto_fixed", False)])

"engine_stats": {
    "quick_path_usage": quick_path_count / len(successful_results) if successful_results else 0,
    "auto_fixed_count": auto_fixed_count,
    "auto_fixed_rate": auto_fixed_count / len(successful_results) if successful_results else 0
}
```

#### 6.2 增强的性能统计

```python
"performance_stats": {
    "inference_time": {...},
    "confidence": {
        "mean": np.mean(confidences),
        "std": np.std(confidences),
        "min": np.min(confidences),
        "max": np.max(confidences)
    }
}
```

### 7. 配置保存扩展

```python
"config": {
    "ai_model": self.config.ai_config.model_path,
    "temperature": self.config.ai_config.temperature,
    "confidence_threshold": self.config.ai_config.confidence_threshold,
    "enable_pattern_cache": self.config.ai_config.enable_pattern_cache,
    "enable_knowledge_base": self.config.ai_config.enable_knowledge_base,
    "enable_semantic_analysis": self.config.ai_config.enable_semantic_analysis,
    "max_samples_per_file": self.config.max_samples_per_file,
    "max_files_per_type": self.config.max_files_per_type
}
```

### 8. 命令行参数扩展

```python
parser.add_argument("--model-path", default="/data/LLM_models/Qwen2.5-Coder-14B-Instruct", help="AI模型路径")
parser.add_argument("--confidence-threshold", type=float, default=0.6, help="置信度阈值")
parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
```

### 9. 输出报告增强

#### 9.1 新增引擎特性指标

```python
if read_ops.get("engine_stats"):
    engine_stats = read_ops["engine_stats"]
    print(f"快速路径使用率: {engine_stats['quick_path_usage']:.2%}")
    print(f"自动修复率: {engine_stats['auto_fixed_rate']:.2%}")
```

#### 9.2 置信度统计

```python
print(f"平均置信度: {perf_stats['confidence']['mean']:.3f}")
```

#### 9.3 技术标识

```python
print("注：使用多层级推理引擎 (模式分析→语义理解→知识匹配→LLM生成→验证优化)")
```

## 新增的演示脚本

创建了 `multilevel_inference_demo.py` 演示脚本，具有以下功能：

### 1. 完整的场景演示
- UART数据读取场景
- UART数据写入场景  
- UART状态检查场景

### 2. 性能评估
- 推理时间统计
- 置信度分析
- 快速路径使用率
- 自动修复率

### 3. 结果可视化
- 实时结果打印
- 性能摘要生成
- JSON结果保存

## 兼容性说明

### 向前兼容性
- 支持旧格式和新格式的JSON数据输入
- 自动适配不同的数据结构

### 错误处理增强
- 增加了详细的异常捕获和日志记录
- 提供更友好的错误信息

### 性能优化
- 支持多层级推理的性能优化特性
- 统计和报告优化效果

## 使用方式

### 1. 基本使用
```bash
python3 json_data_processor.py --input-dir ./evaluation_data
```

### 2. 高级配置
```bash
python3 json_data_processor.py \
    --input-dir ./evaluation_data \
    --output-dir ./results \
    --model-path /data/LLM_models/Qwen2.5-Coder-14B-Instruct \
    --confidence-threshold 0.7 \
    --temperature 0.1 \
    --max-samples 100
```

### 3. 演示脚本
```bash
python3 multilevel_inference_demo.py
```

## 技术优势

### 1. 架构升级
- 从单层推理升级到多层级推理
- 支持模式分析、语义理解、知识匹配等高级功能

### 2. 准确性提升
- 多维度置信度计算
- 自动错误修复机制
- 智能约束验证

### 3. 性能优化
- 快速路径支持
- 模式缓存机制
- 知识库加速

### 4. 可扩展性
- 支持多种外设类型
- 可配置的约束系统
- 标准化的接口设计

## 结论

通过这次全面更新，`json_data_processor.py` 已经完全适配了重构后的多层级AI推理引擎，不仅保持了原有功能，还新增了许多高级特性，为AI推理评估提供了更强大、更精确的工具。 
