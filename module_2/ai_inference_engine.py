#!/usr/bin/env python3
"""
通用AI推理引擎 - 支持多层级推断策略
优先保证准确性，同时优化推理速度
"""

import json
import time
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
import hashlib
from collections import defaultdict, deque
import sqlite3
import pickle

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PeripheralType(Enum):
    """外设类型枚举"""
    UART = "UART"
    SPI = "SPI"
    I2C = "I2C"
    GPIO = "GPIO"
    TIMER = "TIMER"
    DMA = "DMA"
    ETHERNET = "ETHERNET"
    USB = "USB"
    UNKNOWN = "UNKNOWN"

class AccessType(Enum):
    """访问类型枚举"""
    READ = "READ"
    WRITE = "WRITE"
    MODIFY = "MODIFY"

@dataclass
class DeviceConfig:
    """设备配置定义"""
    device_type: PeripheralType
    device_name: str
    base_address: int
    register_map: Dict[int, str]  # offset -> register_name
    critical_registers: Set[str]
    register_constraints: Dict[str, Dict[str, Any]]  # register -> constraints
    behavioral_patterns: Dict[str, Any]  # 行为模式定义
    
    def __post_init__(self):
        if not self.critical_registers:
            self.critical_registers = set()
        if not self.register_constraints:
            self.register_constraints = {}
        if not self.behavioral_patterns:
            self.behavioral_patterns = {}

@dataclass
class AccessPattern:
    """访问模式定义"""
    sequence: List[Dict[str, Any]]  # 访问序列
    timestamp: float
    context: Dict[str, Any]  # 上下文信息
    pattern_hash: str = field(init=False)
    
    def __post_init__(self):
        # 生成模式哈希用于快速匹配
        pattern_str = json.dumps(self.sequence, sort_keys=True)
        self.pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()

@dataclass
class InferenceConstraints:
    """推理约束定义"""
    min_registers_count: int = 3
    max_registers_count: int = 20
    register_value_range: Tuple[int, int] = (0x0, 0xFFFFFFFF)
    required_registers: Set[str] = field(default_factory=set)
    forbidden_values: Dict[str, Set[int]] = field(default_factory=dict)
    logical_constraints: List[str] = field(default_factory=list)  # 逻辑约束表达式
    
    def validate_prediction(self, prediction: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证预测结果是否满足约束"""
        errors = []
        
        if "registers" in prediction:
            registers = prediction["registers"]
            
            # 检查寄存器数量
            reg_count = len(registers)
            if reg_count < self.min_registers_count:
                errors.append(f"寄存器数量不足: {reg_count} < {self.min_registers_count}")
            elif reg_count > self.max_registers_count:
                errors.append(f"寄存器数量过多: {reg_count} > {self.max_registers_count}")
            
            # 检查必需寄存器
            missing_reqs = self.required_registers - set(registers.keys())
            if missing_reqs:
                errors.append(f"缺少必需寄存器: {missing_reqs}")
            
            # 检查寄存器值范围和禁止值
            for reg_name, reg_value in registers.items():
                try:
                    val = int(reg_value, 16) if isinstance(reg_value, str) else reg_value
                    
                    # 范围检查
                    if not (self.register_value_range[0] <= val <= self.register_value_range[1]):
                        errors.append(f"寄存器{reg_name}值超出范围: {hex(val)}")
                    
                    # 禁止值检查
                    if reg_name in self.forbidden_values and val in self.forbidden_values[reg_name]:
                        errors.append(f"寄存器{reg_name}使用了禁止值: {hex(val)}")
                        
                except ValueError:
                    errors.append(f"寄存器{reg_name}值格式无效: {reg_value}")
        
        return len(errors) == 0, errors

@dataclass
class InferenceConfig:
    """通用推理配置"""
    model_path: str = "/data/LLM_models/Qwen2.5-Coder-14B-Instruct"
    max_length: int = 4096
    temperature: float = 0.2  # 降低温度提高一致性
    top_p: float = 0.95
    do_sample: bool = True
    
    # 性能配置
    load_in_8bit: bool = False
    use_flash_attention: bool = False
    
    # 多层级推理配置
    enable_pattern_cache: bool = True
    enable_knowledge_base: bool = True
    enable_semantic_analysis: bool = True
    max_cache_size: int = 10000
    
    # 推理约束
    constraints: InferenceConstraints = field(default_factory=InferenceConstraints)
    
    # 推理策略
    max_retries: int = 3
    timeout_seconds: int = 60
    confidence_threshold: float = 0.8

class PatternAnalyzer:
    """MMIO访问模式分析器"""
    
    def __init__(self):
        self.common_patterns = {}
        self.sequence_cache = {}
        
    def analyze_mmio_sequence(self, access_pattern: AccessPattern) -> Dict[str, Any]:
        """分析MMIO访问序列"""
        sequence = access_pattern.sequence
        
        # 快速模式匹配
        if access_pattern.pattern_hash in self.sequence_cache:
            return self.sequence_cache[access_pattern.pattern_hash]
        
        analysis = {
            "pattern_type": self._identify_pattern_type(sequence),
            "access_frequency": self._calculate_access_frequency(sequence),
            "register_dependencies": self._find_register_dependencies(sequence),
            "temporal_patterns": self._analyze_temporal_patterns(sequence),
            "confidence": 0.0
        }
        
        # 计算分析置信度
        analysis["confidence"] = self._calculate_pattern_confidence(analysis)
        
        # 缓存结果
        if len(self.sequence_cache) < 1000:  # 限制缓存大小
            self.sequence_cache[access_pattern.pattern_hash] = analysis
        
        return analysis
    
    def _identify_pattern_type(self, sequence: List[Dict[str, Any]]) -> str:
        """识别访问模式类型"""
        if not sequence:
            return "empty"
        
        # 分析访问类型分布
        read_count = sum(1 for op in sequence if op.get("type") == "READ")
        write_count = sum(1 for op in sequence if op.get("type") == "WRITE")
        
        if read_count > write_count * 2:
            return "read_heavy"
        elif write_count > read_count * 2:
            return "write_heavy"
        elif len(sequence) == 1:
            return "single_access"
        else:
            return "mixed_access"
    
    def _calculate_access_frequency(self, sequence: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算寄存器访问频率"""
        frequency = defaultdict(int)
        for op in sequence:
            reg_name = op.get("register", "unknown")
            frequency[reg_name] += 1
        return dict(frequency)
    
    def _find_register_dependencies(self, sequence: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """查找寄存器依赖关系"""
        dependencies = []
        for i in range(len(sequence) - 1):
            current_reg = sequence[i].get("register")
            next_reg = sequence[i + 1].get("register")
            if current_reg and next_reg and current_reg != next_reg:
                dependencies.append((current_reg, next_reg))
        return dependencies
    
    def _analyze_temporal_patterns(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间模式"""
        if len(sequence) < 2:
            return {"intervals": [], "regularity": 0.0}
        
        intervals = []
        for i in range(len(sequence) - 1):
            t1 = sequence[i].get("timestamp", 0)
            t2 = sequence[i + 1].get("timestamp", 0)
            intervals.append(t2 - t1)
        
        # 计算规律性（方差越小越规律）
        if intervals:
            mean_interval = np.mean(intervals)
            variance = np.var(intervals)
            regularity = 1.0 / (1.0 + variance) if mean_interval > 0 else 0.0
        else:
            regularity = 0.0
        
        return {
            "intervals": intervals,
            "mean_interval": np.mean(intervals) if intervals else 0,
            "regularity": regularity
        }
    
    def _calculate_pattern_confidence(self, analysis: Dict[str, Any]) -> float:
        """计算模式分析置信度"""
        confidence = 0.5  # 基础置信度
        
        # 根据模式类型调整置信度
        pattern_type = analysis.get("pattern_type", "unknown")
        if pattern_type in ["read_heavy", "write_heavy", "single_access"]:
            confidence += 0.2
        
        # 根据时间规律性调整
        temporal = analysis.get("temporal_patterns", {})
        regularity = temporal.get("regularity", 0)
        confidence += regularity * 0.3
        
        return min(confidence, 1.0)

class SemanticAnalyzer:
    """语义分析器 - 理解驱动行为意图"""
    
    def __init__(self):
        self.behavior_patterns = self._load_behavior_patterns()
    
    def understand_driver_behavior(self, pattern_analysis: Dict[str, Any], 
                                 device_config: DeviceConfig) -> Dict[str, Any]:
        """理解驱动程序行为意图"""
        pattern_type = pattern_analysis.get("pattern_type", "unknown")
        device_type = device_config.device_type
        
        # 基于设备类型和访问模式推断意图
        intent = {
            "operation_intent": self._infer_operation_intent(pattern_type, device_type),
            "driver_state": self._infer_driver_state(pattern_analysis, device_config),
            "expected_outcome": self._predict_expected_outcome(pattern_analysis, device_config),
            "confidence": 0.0
        }
        
        # 计算语义分析置信度
        intent["confidence"] = self._calculate_semantic_confidence(intent, pattern_analysis)
        
        return intent
    
    def _infer_operation_intent(self, pattern_type: str, device_type: PeripheralType) -> str:
        """推断操作意图"""
        intent_map = {
            (PeripheralType.UART, "read_heavy"): "data_reception",
            (PeripheralType.UART, "write_heavy"): "data_transmission", 
            (PeripheralType.UART, "single_access"): "status_check",
            (PeripheralType.SPI, "write_heavy"): "spi_transfer",
            (PeripheralType.I2C, "mixed_access"): "i2c_transaction",
            (PeripheralType.GPIO, "single_access"): "pin_control",
        }
        
        return intent_map.get((device_type, pattern_type), "generic_operation")
    
    def _infer_driver_state(self, pattern_analysis: Dict[str, Any], 
                           device_config: DeviceConfig) -> str:
        """推断驱动程序状态"""
        frequency = pattern_analysis.get("access_frequency", {})
        
        # 检查关键寄存器访问
        critical_accessed = any(reg in frequency for reg in device_config.critical_registers)
        
        if critical_accessed:
            return "active_operation"
        elif len(frequency) == 1:
            return "simple_access"
        else:
            return "complex_operation"
    
    def _predict_expected_outcome(self, pattern_analysis: Dict[str, Any], 
                                device_config: DeviceConfig) -> Dict[str, Any]:
        """预测期望结果"""
        pattern_type = pattern_analysis.get("pattern_type")
        
        outcome = {
            "register_changes": [],
            "side_effects": [],
            "state_transitions": []
        }
        
        # 基于设备行为模式预测
        if device_config.behavioral_patterns:
            for pattern_name, pattern_def in device_config.behavioral_patterns.items():
                if self._pattern_matches(pattern_analysis, pattern_def):
                    outcome["register_changes"].extend(pattern_def.get("register_changes", []))
                    outcome["side_effects"].extend(pattern_def.get("side_effects", []))
        
        return outcome
    
    def _pattern_matches(self, analysis: Dict[str, Any], pattern_def: Dict[str, Any]) -> bool:
        """检查模式是否匹配"""
        # 简化的模式匹配逻辑
        required_type = pattern_def.get("pattern_type")
        if required_type and analysis.get("pattern_type") != required_type:
            return False
        return True
    
    def _calculate_semantic_confidence(self, intent: Dict[str, Any], 
                                     pattern_analysis: Dict[str, Any]) -> float:
        """计算语义分析置信度"""
        base_confidence = pattern_analysis.get("confidence", 0.5)
        
        # 根据意图明确性调整
        operation_intent = intent.get("operation_intent", "generic_operation")
        if operation_intent != "generic_operation":
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _load_behavior_patterns(self) -> Dict[str, Any]:
        """加载行为模式定义"""
        return {
            "uart_init": {
                "pattern_type": "write_heavy",
                "register_changes": ["control_reg", "baud_rate_reg"],
                "side_effects": ["fifo_reset", "interrupt_setup"]
            },
            "uart_transmit": {
                "pattern_type": "write_heavy", 
                "register_changes": ["data_reg", "status_reg"],
                "side_effects": ["fifo_update", "interrupt_trigger"]
            }
        }

class KnowledgeBase:
    """设备知识库"""
    
    def __init__(self, db_path: str = "device_knowledge.db"):
        self.db_path = db_path
        self.device_configs = {}
        self.pattern_cache = {}
        self._init_database()
        self._load_device_configs()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建表结构
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS device_patterns (
                    pattern_hash TEXT PRIMARY KEY,
                    device_type TEXT,
                    pattern_data BLOB,
                    confidence REAL,
                    usage_count INTEGER,
                    last_used TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS device_configs (
                    device_name TEXT PRIMARY KEY,
                    device_type TEXT,
                    config_data BLOB
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"数据库初始化失败: {e}")
    
    def search_device_database(self, pattern_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """在设备数据库中搜索匹配模式"""
        pattern_type = pattern_analysis.get("pattern_type")
        confidence = pattern_analysis.get("confidence", 0.0)
        
        # 快速内存缓存查找
        cache_key = f"{pattern_type}_{confidence:.2f}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        # 数据库查找
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pattern_data, confidence FROM device_patterns 
                WHERE device_type = ? AND confidence > ?
                ORDER BY confidence DESC, usage_count DESC
                LIMIT 5
            ''', (pattern_type, confidence * 0.8))
            
            results = cursor.fetchall()
            conn.close()
            
            if results:
                # 返回最佳匹配
                best_match = pickle.loads(results[0][0])
                
                # 缓存结果
                if len(self.pattern_cache) < 100:
                    self.pattern_cache[cache_key] = best_match
                
                return best_match
                
        except Exception as e:
            logger.warning(f"数据库查询失败: {e}")
        
        return None
    
    def get_device_config(self, device_name: str) -> Optional[DeviceConfig]:
        """获取设备配置"""
        if device_name in self.device_configs:
            return self.device_configs[device_name]
        
        # 从数据库加载
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT config_data FROM device_configs WHERE device_name = ?', 
                         (device_name,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                config = pickle.loads(result[0])
                self.device_configs[device_name] = config
                return config
                
        except Exception as e:
            logger.warning(f"设备配置加载失败: {e}")
        
        return None
    
    def store_pattern(self, pattern_hash: str, device_type: str, 
                     pattern_data: Dict[str, Any], confidence: float):
        """存储模式到知识库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO device_patterns 
                (pattern_hash, device_type, pattern_data, confidence, usage_count, last_used)
                VALUES (?, ?, ?, ?, 1, datetime('now'))
            ''', (pattern_hash, device_type, pickle.dumps(pattern_data), confidence))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"模式存储失败: {e}")
    
    def _load_device_configs(self):
        """加载预定义设备配置"""
        # PL011 UART配置
        pl011_config = DeviceConfig(
            device_type=PeripheralType.UART,
            device_name="PL011",
            base_address=0x09000000,
            register_map={
                0x00: "UARTDR",
                0x04: "UARTRSR", 
                0x18: "UARTFR",
                0x24: "UARTIBRD",
                0x28: "UARTFBRD", 
                0x2C: "UARTLCR_H",
                0x30: "UARTCR",
                0x34: "UARTIFLS",
                0x38: "UARTIMSC",
                0x3C: "UARTRIS",
                0x40: "UARTMIS",
                0x48: "UARTDMACR"
            },
            critical_registers={"UARTDR", "UARTFR", "UARTCR", "UARTIMSC"},
            register_constraints={
                "UARTDR": {"range": (0x0, 0xFF), "writable": True},
                "UARTFR": {"range": (0x0, 0x1FF), "writable": False},
                "UARTCR": {"range": (0x0, 0xFFFF), "writable": True}
            },
            behavioral_patterns={
                "fifo_read": {
                    "pattern_type": "read_heavy",
                    "trigger_register": "UARTDR",
                    "affected_registers": ["UARTFR", "UARTRIS"],
                    "side_effects": ["fifo_pop", "flag_update"]
                },
                "fifo_write": {
                    "pattern_type": "write_heavy",
                    "trigger_register": "UARTDR", 
                    "affected_registers": ["UARTFR", "UARTRIS"],
                    "side_effects": ["fifo_push", "flag_update"]
                }
            }
        )
        
        self.device_configs["PL011"] = pl011_config

class UniversalPromptEngine:
    """通用提示词生成引擎"""
    
    def __init__(self, device_config: DeviceConfig):
        self.device_config = device_config
    
    def create_prompt(self, sample_data: Dict[str, Any], 
                     semantic_intent: Dict[str, Any] = None,
                     knowledge_match: Dict[str, Any] = None) -> str:
        """创建智能提示词"""
        
        # 提取操作信息
        operation_info = self._extract_operation_info(sample_data)
        
        # 构建基础提示
        prompt = self._build_base_prompt(operation_info)
        
        # 添加设备特定信息
        prompt += self._add_device_context(operation_info)
        
        # 添加语义意图信息
        if semantic_intent:
            prompt += self._add_semantic_context(semantic_intent)
        
        # 添加知识库匹配信息
        if knowledge_match:
            prompt += self._add_knowledge_context(knowledge_match)
        
        # 添加约束和格式要求
        prompt += self._add_constraints_and_format()
        
        return prompt
    
    def _extract_operation_info(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取操作信息"""
        operation = sample_data.get("operation", "")
        
        # 通用的操作解析
        info = {
            "type": AccessType.READ if "READ" in operation.upper() else AccessType.WRITE,
            "offset": self._extract_offset(operation),
            "value": self._extract_value(operation) if "WRITE" in operation.upper() else None,
            "current_registers": sample_data.get("current_registers", {})
        }
        
        return info
    
    def _extract_offset(self, operation: str) -> str:
        """提取偏移量"""
        match = re.search(r'offset=(0x[0-9a-fA-F]+)', operation)
        return match.group(1) if match else "unknown"
    
    def _extract_value(self, operation: str) -> Optional[str]:
        """提取写入值"""
        match = re.search(r'value=(0x[0-9a-fA-F]+)', operation)
        return match.group(1) if match else None
    
    def _build_base_prompt(self, operation_info: Dict[str, Any]) -> str:
        """构建基础提示"""
        device_type = self.device_config.device_type.value
        device_name = self.device_config.device_name
        
        prompt = f"""You are an expert in {device_type} peripheral hardware simulation. 
I need you to predict the behavior of a {device_name} {device_type} peripheral operation with high accuracy.

## Operation Details
- **Device**: {device_name} {device_type}
- **Operation Type**: {operation_info['type'].value}
- **Register Offset**: {operation_info['offset']}"""
        
        if operation_info['value']:
            prompt += f"\n- **Write Value**: {operation_info['value']}"
        
        return prompt
    
    def _add_device_context(self, operation_info: Dict[str, Any]) -> str:
        """添加设备上下文"""
        context = f"""

## Current Register State
"""
        for reg_name, reg_value in sorted(operation_info['current_registers'].items()):
            context += f"- {reg_name.upper()}: {reg_value}\n"
        
        context += f"""
## Device Register Map
"""
        for offset, reg_name in sorted(self.device_config.register_map.items()):
            context += f"- 0x{offset:02X}: {reg_name}\n"
        
        return context
    
    def _add_semantic_context(self, semantic_intent: Dict[str, Any]) -> str:
        """添加语义上下文"""
        return f"""
## Semantic Analysis
- **Operation Intent**: {semantic_intent.get('operation_intent', 'unknown')}
- **Driver State**: {semantic_intent.get('driver_state', 'unknown')}
- **Analysis Confidence**: {semantic_intent.get('confidence', 0.0):.2f}
"""
    
    def _add_knowledge_context(self, knowledge_match: Dict[str, Any]) -> str:
        """添加知识库上下文"""
        return f"""
## Knowledge Base Match
- **Pattern Found**: {knowledge_match.get('pattern_type', 'none')}
- **Expected Behavior**: {knowledge_match.get('expected_behavior', 'standard')}
- **Match Confidence**: {knowledge_match.get('confidence', 0.0):.2f}
"""
    
    def _add_constraints_and_format(self) -> str:
        """添加约束和格式要求"""
        critical_regs = ", ".join(self.device_config.critical_registers)
        
        return f"""
## Critical Requirements
- **Critical Registers**: {critical_regs}
- **Register Count**: Must predict at least 3 registers
- **Value Format**: All values must be valid hexadecimal (0x format)
- **Accuracy**: Predict EXACT values, not placeholders

## Response Format
Provide your prediction in this EXACT JSON format:

```json
{{
  "read_value": "0x90",
  "registers": {{
    "register1": "0x00000000",
    "register2": "0x00000090"
  }}
}}
```

CRITICAL: Use actual hex values, never placeholder text. For WRITE operations, set "read_value" to null.

Your response:"""

class MultiLevelInferenceEngine:
    """多层级推理引擎"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.pattern_analyzer = PatternAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.knowledge_base = KnowledgeBase()
        
        # 加载AI模型（延迟加载）
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
    
    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if not self._model_loaded:
            self._load_ai_model()
            self._model_loaded = True
    
    def _load_ai_model(self):
        """加载AI模型"""
        logger.info(f"加载AI模型: {self.config.model_path}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }
            
            if self.config.load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            self.model.eval()
            logger.info("AI模型加载完成")
            
        except Exception as e:
            logger.error(f"AI模型加载失败: {e}")
            raise
    
    def intelligent_inference(self, access_pattern: AccessPattern, 
                            device_name: str = "PL011") -> Dict[str, Any]:
        """智能多层级推理"""
        start_time = time.time()
        results = {}
        
        try:
            # 获取设备配置
            device_config = self.knowledge_base.get_device_config(device_name)
            if not device_config:
                raise ValueError(f"未找到设备配置: {device_name}")
            
            # 第一层：模式识别
            logger.info("第一层：MMIO访问模式识别")
            pattern_analysis = self.pattern_analyzer.analyze_mmio_sequence(access_pattern)
            results["pattern_analysis"] = pattern_analysis
            
            # 如果模式置信度很高，尝试快速路径
            if pattern_analysis.get("confidence", 0) > 0.9:
                quick_result = self._try_quick_inference(pattern_analysis, device_config)
                if quick_result:
                    results["quick_path"] = True
                    results["prediction"] = quick_result
                    results["inference_time"] = time.time() - start_time
                    return results
            
            # 第二层：语义理解
            logger.info("第二层：驱动行为语义理解")
            semantic_intent = self.semantic_analyzer.understand_driver_behavior(
                pattern_analysis, device_config
            )
            results["semantic_intent"] = semantic_intent
            
            # 第三层：知识库匹配
            logger.info("第三层：设备知识库匹配")
            knowledge_match = self.knowledge_base.search_device_database(pattern_analysis)
            results["knowledge_match"] = knowledge_match
            
            # 第四层：动态LLM生成
            logger.info("第四层：动态LLM推理生成")
            llm_prediction = self._generate_with_llm(
                access_pattern, device_config, semantic_intent, knowledge_match
            )
            results["llm_prediction"] = llm_prediction
            
            # 验证和后处理
            final_prediction = self._validate_and_refine(
                llm_prediction, device_config
            )
            results["prediction"] = final_prediction
            
            # 更新知识库
            if final_prediction and pattern_analysis.get("confidence", 0) > 0.7:
                self._update_knowledge_base(access_pattern, final_prediction, device_config)
            
            results["inference_time"] = time.time() - start_time
            return results
            
        except Exception as e:
            logger.error(f"智能推理失败: {e}")
            results["error"] = str(e)
            results["inference_time"] = time.time() - start_time
            return results
    
    def _try_quick_inference(self, pattern_analysis: Dict[str, Any], 
                           device_config: DeviceConfig) -> Optional[Dict[str, Any]]:
        """尝试快速推理路径"""
        pattern_type = pattern_analysis.get("pattern_type")
        
        # 对于简单的单次访问，使用预定义规则
        if pattern_type == "single_access":
            # 这里可以实现基于规则的快速推理
            # 例如：读取状态寄存器的标准响应
            pass
        
        return None
    
    def _generate_with_llm(self, access_pattern: AccessPattern,
                          device_config: DeviceConfig,
                          semantic_intent: Dict[str, Any] = None,
                          knowledge_match: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """使用LLM生成预测"""
        self._ensure_model_loaded()
        
        # 使用最后一个访问作为样本数据
        if not access_pattern.sequence:
            return None
        
        sample_data = access_pattern.sequence[-1]
        
        # 创建智能提示词
        prompt_engine = UniversalPromptEngine(device_config)
        prompt = prompt_engine.create_prompt(sample_data, semantic_intent, knowledge_match)
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.model.device)
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            # 解码响应
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # 解析响应
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM响应"""
        try:
            # 寻找JSON块
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # 尝试直接解析
            try:
                return json.loads(response)
            except:
                pass
            
            # 手动解析关键信息
            result = {}
            
            # 解析读取值
            read_match = re.search(r'"read_value"\s*:\s*"(0x[0-9a-fA-F]+)"', response)
            if read_match:
                result["read_value"] = read_match.group(1)
            
            # 解析寄存器
            registers = {}
            reg_pattern = r'"(\w+)"\s*:\s*"(0x[0-9a-fA-F]+)"'
            for match in re.finditer(reg_pattern, response):
                reg_name, reg_value = match.groups()
                if reg_name != "read_value":
                    registers[reg_name] = reg_value
            
            if registers:
                result["registers"] = registers
            
            return result if result else None
            
        except Exception as e:
            logger.error(f"响应解析失败: {e}")
            return None
    
    def _validate_and_refine(self, prediction: Dict[str, Any], 
                           device_config: DeviceConfig) -> Dict[str, Any]:
        """验证和优化预测结果"""
        if not prediction:
            return {"error": "无预测结果"}
        
        # 应用约束验证
        is_valid, errors = self.config.constraints.validate_prediction(prediction)
        
        result = {
            "prediction": prediction,
            "validation_passed": is_valid,
            "validation_errors": errors,
            "confidence": self._calculate_overall_confidence(prediction, device_config)
        }
        
        # 如果验证失败但错误不严重，尝试修复
        if not is_valid and len(errors) <= 2:
            fixed_prediction = self._attempt_fix(prediction, errors, device_config)
            if fixed_prediction:
                result["prediction"] = fixed_prediction
                result["validation_passed"] = True
                result["auto_fixed"] = True
        
        return result
    
    def _calculate_overall_confidence(self, prediction: Dict[str, Any], 
                                    device_config: DeviceConfig) -> float:
        """计算整体置信度"""
        confidence = 0.5  # 基础置信度
        
        # 根据寄存器数量调整
        if "registers" in prediction:
            reg_count = len(prediction["registers"])
            if reg_count >= len(device_config.critical_registers):
                confidence += 0.2
        
        # 根据关键寄存器覆盖度调整
        if "registers" in prediction:
            covered_critical = set(prediction["registers"].keys()) & device_config.critical_registers
            coverage_ratio = len(covered_critical) / len(device_config.critical_registers)
            confidence += coverage_ratio * 0.3
        
        return min(confidence, 1.0)
    
    def _attempt_fix(self, prediction: Dict[str, Any], errors: List[str], 
                    device_config: DeviceConfig) -> Optional[Dict[str, Any]]:
        """尝试修复预测结果"""
        fixed = prediction.copy()
        
        # 修复缺少的关键寄存器
        if "registers" in fixed:
            for error in errors:
                if "缺少必需寄存器" in error:
                    # 添加默认值
                    missing_regs = device_config.critical_registers - set(fixed["registers"].keys())
                    for reg in missing_regs:
                        fixed["registers"][reg] = "0x00000000"  # 默认值
        
        return fixed
    
    def _update_knowledge_base(self, access_pattern: AccessPattern, 
                             prediction: Dict[str, Any], device_config: DeviceConfig):
        """更新知识库"""
        try:
            pattern_data = {
                "prediction": prediction,
                "device_config": device_config.device_name,
                "timestamp": time.time()
            }
            
            confidence = prediction.get("confidence", 0.5)
            self.knowledge_base.store_pattern(
                access_pattern.pattern_hash,
                device_config.device_type.value,
                pattern_data,
                confidence
            )
        except Exception as e:
            logger.warning(f"知识库更新失败: {e}")

def main():
    """测试多层级推理引擎"""
    # 配置
    config = InferenceConfig()
    config.constraints.required_registers = {"uartfr", "uartcr"}
    config.constraints.min_registers_count = 3
    
    # 创建引擎
    engine = MultiLevelInferenceEngine(config)
    
    # 测试访问模式
    access_sequence = [
        {
            "type": "READ",
            "register": "uartfr",
            "offset": "0x18",
            "timestamp": time.time(),
            "operation": "PL011 UART READ offset=0x18",
            "current_registers": {
                "uartdr": "0x00000000",
                "uartfr": "0x00000090",
                "uartcr": "0x00000f01",
                "uartibrd": "0x00000027",
                "uartfbrd": "0x00000004",
                "uartlcr_h": "0x00000070",
                "uartimsc": "0x00000050"
            }
        }
    ]
    
    access_pattern = AccessPattern(
        sequence=access_sequence,
        timestamp=time.time(),
        context={"device": "PL011", "driver": "uart_driver"}
    )
    
    print("开始多层级智能推理...")
    result = engine.intelligent_inference(access_pattern, "PL011")
    
    print(f"推理结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    main() 
