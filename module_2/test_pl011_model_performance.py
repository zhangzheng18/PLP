#!/usr/bin/env python3
"""
PL011外设AI模型性能测试系统
使用先进的推理技术和全面的评估指标测试模型效果
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import re
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """推理配置"""
    # 模型配置
    base_model_name: str = "/data/LLM_models/Qwen2.5-Coder-14B-Instruct"
    # base_model_name: str = "/data/vllm_models/qwq-32bt"
    finetuned_model_path: str = "./pl011_finetuned_model"
    use_finetuned: bool = False  # 默认不使用微调模型
    
    # 推理参数
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 3
    repetition_penalty: float = 1.1
    
    # 集成推理配置
    enable_ensemble: bool = False  # 默认不启用集成推理，先测试单次推理
    ensemble_size: int = 3
    confidence_threshold: float = 0.7
    timeout_seconds: int = 30
    
    # 评估配置
    test_data_dir: str = "./evaluation_data"
    output_dir: str = "./evaluation_results"
    max_test_samples_per_type: int = 20  # 减少样本数量以快速测试
    test_files_per_type: int = 1  # 只使用1个文件进行测试
    
    # 新增：推理约束配置
    min_required_registers: int = 3  # 最少需要推理的寄存器数量
    max_register_value: str = "0xFFFFFFFF"  # 寄存器值上限
    min_register_value: str = "0x00000000"  # 寄存器值下限
    require_return_value: bool = True  # 是否必须有返回值（对于读操作）
    enforce_format_constraints: bool = True  # 是否强制格式约束

@dataclass
class PeripheralConfig:
    """外设配置 - 支持不同类型的外设"""
    peripheral_type: str = "UART"  # 外设类型：UART, SPI, I2C, GPIO等
    peripheral_name: str = "PL011"  # 具体外设名称
    base_address: str = "0x09000000"  # 基地址
    address_space_size: str = "0x1000"  # 地址空间大小
    
    # 寄存器配置
    register_definitions: Dict[str, Dict] = None  # 寄存器定义
    critical_registers: List[str] = None  # 关键寄存器列表
    
    def __post_init__(self):
        if self.register_definitions is None:
            # PL011 UART 寄存器定义
            self.register_definitions = {
                "UARTDR": {"offset": "0x000", "description": "Data Register", "type": "data", "fifo_related": True},
                "UARTRSR": {"offset": "0x004", "description": "Receive Status Register", "type": "status", "fifo_related": False},
                "UARTFR": {"offset": "0x018", "description": "Flag Register", "type": "status", "fifo_related": True},
                "UARTIBRD": {"offset": "0x024", "description": "Integer Baud Rate Divisor", "type": "config", "fifo_related": False},
                "UARTFBRD": {"offset": "0x028", "description": "Fractional Baud Rate Divisor", "type": "config", "fifo_related": False},
                "UARTLCR_H": {"offset": "0x02C", "description": "Line Control Register", "type": "control", "fifo_related": False},
                "UARTCR": {"offset": "0x030", "description": "Control Register", "type": "control", "fifo_related": False},
                "UARTIFLS": {"offset": "0x034", "description": "Interrupt FIFO Level Select", "type": "control", "fifo_related": True},
                "UARTIMSC": {"offset": "0x038", "description": "Interrupt Mask Set/Clear", "type": "interrupt", "fifo_related": False},
                "UARTRIS": {"offset": "0x03C", "description": "Raw Interrupt Status", "type": "interrupt", "fifo_related": True},
                "UARTMIS": {"offset": "0x040", "description": "Masked Interrupt Status", "type": "interrupt", "fifo_related": True},
                "UARTDMACR": {"offset": "0x048", "description": "DMA Control Register", "type": "control", "fifo_related": False}
            }
        
        if self.critical_registers is None:
            # 关键寄存器：数据寄存器、状态寄存器、控制寄存器
            self.critical_registers = ["UARTDR", "UARTFR", "UARTCR", "UARTIMSC", "UARTRIS", "UARTMIS"]

class RegisterParser:
    """通用寄存器解析器"""
    
    def __init__(self, peripheral_config: PeripheralConfig):
        self.peripheral_config = peripheral_config
        self.register_patterns = self._build_register_patterns()
        self.read_value_pattern = r'(?:Read Value|Return Value|Result):\s*0x([0-9a-fA-F]+)'
        self.write_return_pattern = r'(?:Write Return|Write Result):\s*0x([0-9a-fA-F]+)'
    
    def _build_register_patterns(self) -> Dict[str, str]:
        """根据外设配置构建寄存器匹配模式"""
        patterns = {}
        for reg_name in self.peripheral_config.register_definitions.keys():
            patterns[reg_name] = f'{reg_name}=0x([0-9a-fA-F]+)'
        return patterns
    
    def parse_registers(self, text: str) -> Dict[str, str]:
        """从文本中解析寄存器值"""
        registers = {}
        for reg_name, pattern in self.register_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                registers[reg_name.lower()] = f"0x{match.group(1).lower()}"
        return registers
    
    def parse_read_value(self, text: str) -> Optional[str]:
        """从文本中解析读取值"""
        match = re.search(self.read_value_pattern, text, re.IGNORECASE)
        if match:
            return f"0x{match.group(1).lower()}"
        return None
    
    def parse_write_return(self, text: str) -> Optional[str]:
        """从文本中解析写操作返回值"""
        match = re.search(self.write_return_pattern, text, re.IGNORECASE)
        if match:
            return f"0x{match.group(1).lower()}"
        return None

class UniversalPromptEngine:
    """通用外设推理提示词引擎"""
    
    def __init__(self, peripheral_config: PeripheralConfig, inference_config: InferenceConfig):
        self.peripheral_config = peripheral_config
        self.inference_config = inference_config
        
        # 动态生成bit字段分析（可扩展到其他外设）
        self.bit_field_analysis = self._build_bit_field_analysis()
        
        # 推理约束
        self.constraints = self._build_inference_constraints()
    
    def _build_bit_field_analysis(self) -> Dict[str, Dict[str, str]]:
        """构建位字段分析，支持不同外设类型"""
        if self.peripheral_config.peripheral_type == "UART":
            return {
                "UARTFR": {
                    "0x90": "TXFE(1)+RXFE(1): TX FIFO Empty + RX FIFO Empty",
                    "0x80": "TXFE(1): TX FIFO Empty",
                    "0x10": "RXFE(1): RX FIFO Empty",
                    "0x20": "TXFF(1): TX FIFO Full",
                    "0x40": "RXFF(1): RX FIFO Full"
                },
                "UARTCR": {
                    "0x300": "TXE(1)+RXE(1): TX Enable + RX Enable",
                    "0x301": "TXE(1)+RXE(1)+UARTEN(1): TX Enable + RX Enable + UART Enable",
                    "0x001": "UARTEN(1): UART Enable Only"
                }
            }
        else:
            # 为其他外设类型预留扩展空间
            return {}
    
    def _build_inference_constraints(self) -> Dict[str, Any]:
        """构建推理约束"""
        return {
            "register_value_range": {
                "min": self.inference_config.min_register_value,
                "max": self.inference_config.max_register_value
            },
            "minimum_registers_required": self.inference_config.min_required_registers,
            "critical_registers": self.peripheral_config.critical_registers,
            "format_requirements": {
                "hex_format": True,
                "uppercase_registers": True,
                "exact_precision": True
            }
        }
    
    def create_enhanced_prompt(self, sample: Dict, operation_type: str) -> str:
        """创建通用的增强提示词"""
        
        # 基础模板
        if operation_type == "READ":
            prompt = self._create_universal_read_template()
        else:
            prompt = self._create_universal_write_template()
        
        # 添加外设特定信息
        prompt += self._add_peripheral_specifications()
        
        # 添加推理约束
        prompt += self._add_inference_constraints()
        
        # 添加上下文分析
        prompt += self._add_context_analysis(sample['input'])
        
        # 添加推理指导
        prompt += self._add_reasoning_guidance(operation_type)
        
        # 填充具体数据
        prompt = self._populate_sample_data(prompt, sample)
        
        return prompt
    
    def _create_universal_read_template(self) -> str:
        return f"""You are an expert {self.peripheral_config.peripheral_type} peripheral emulator with deep knowledge of hardware behavior.

### TASK
Predict the result of a READ operation on {self.peripheral_config.peripheral_name} {self.peripheral_config.peripheral_type} peripheral.

### CORE REQUIREMENTS
1. Analyze current register states and system context carefully
2. Apply hardware specifications precisely according to the peripheral type
3. Consider state machine implications and side effects
4. Predict BOTH the read return value AND resulting register state changes
5. Provide exact hexadecimal values within specified constraints

### PERIPHERAL CONTEXT
- Type: {self.peripheral_config.peripheral_type}
- Model: {self.peripheral_config.peripheral_name}
- Base Address: {self.peripheral_config.base_address}
- Address Space: {self.peripheral_config.address_space_size}
"""
    
    def _create_universal_write_template(self) -> str:
        return f"""You are an expert {self.peripheral_config.peripheral_type} peripheral emulator with deep knowledge of hardware behavior.

### TASK
Predict the result of a WRITE operation on {self.peripheral_config.peripheral_name} {self.peripheral_config.peripheral_type} peripheral.

### CORE REQUIREMENTS
1. Analyze the write value and target register functionality
2. Apply register-specific behavior rules and side effects
3. Update ALL affected registers (not just the target register)
4. Consider state transitions and interaction effects
5. Predict write return value (if applicable) AND all resulting register states

### PERIPHERAL CONTEXT
- Type: {self.peripheral_config.peripheral_type}
- Model: {self.peripheral_config.peripheral_name}
- Base Address: {self.peripheral_config.base_address}
- Address Space: {self.peripheral_config.address_space_size}
"""
    
    def _add_peripheral_specifications(self) -> str:
        reg_info = ""
        for reg_name, reg_def in self.peripheral_config.register_definitions.items():
            reg_info += f"- {reg_name} ({reg_def['offset']}): {reg_def['description']} [Type: {reg_def['type']}]\n"
        
        return f"""
### PERIPHERAL SPECIFICATIONS:
{reg_info}

### KEY BEHAVIOR PATTERNS:
- Critical registers that often change together: {', '.join(self.peripheral_config.critical_registers)}
- Register types: data, status, control, interrupt, config
- State dependencies: Status registers reflect data register operations
- Side effects: Control register changes affect status and interrupt registers

"""
    
    def _add_inference_constraints(self) -> str:
        constraints = self.constraints
        critical_regs = ', '.join(constraints["critical_registers"])
        
        return f"""### INFERENCE CONSTRAINTS:
⚠️  MANDATORY REQUIREMENTS:
1. REGISTER COUNT: Must predict at least {constraints["minimum_registers_required"]} register values
2. VALUE RANGE: All register values must be between {constraints["register_value_range"]["min"]} and {constraints["register_value_range"]["max"]}
3. CRITICAL REGISTERS: Must include predictions for key registers: {critical_regs}
4. FORMAT: Use exact hexadecimal format (0x########, lowercase hex digits)
5. COMPLETENESS: For READ operations, must provide both read value AND register changes
6. CAUSALITY: Register changes must be logically consistent with the operation

### VALIDATION RULES:
- All register names must be UPPERCASE
- All hex values must be lowercase with 0x prefix
- Register values must be 32-bit (8 hex digits) unless specified otherwise
- Status registers must reflect logical hardware state
- Interrupt registers must follow enable/disable logic

"""
    
    def _add_context_analysis(self, input_data: Dict) -> str:
        context = "\n### CURRENT SYSTEM CONTEXT:\n"
        
        # CPU状态分析
        cpu_ctx = input_data['cpu_context']
        context += f"- CPU State: PC={cpu_ctx['pc']}, CPU#{cpu_ctx['cpu_index']}\n"
        context += f"- Memory Management: Paging={'ENABLED' if cpu_ctx['paging_enabled'] else 'DISABLED'}\n"
        context += f"- Interrupt Context: IRQ={cpu_ctx['interrupt_request']}\n"
        
        # 访问分析
        context += f"- Memory Access: Offset={input_data['offset']}, Size={input_data['access_size']} bytes\n"
        
        # FIFO分析（如果适用）
        if 'fifo_state' in input_data:
            fifo = input_data['fifo_state']
            context += f"- FIFO Status: Count={fifo['read_count']}, Position={fifo['read_pos']}, Trigger={fifo['read_trigger']}\n"
        
        return context
    
    def _add_reasoning_guidance(self, operation_type: str) -> str:
        if operation_type == "READ":
            return f"""### REASONING METHODOLOGY:
1. TARGET IDENTIFICATION: Determine which register is being read from the offset
2. CURRENT STATE ANALYSIS: Examine all current register values and their relationships
3. OPERATION EFFECTS: Apply read-specific behavior (e.g., FIFO pop, status clear, etc.)
4. CASCADE EFFECTS: Determine which other registers are affected by this read
5. STATE VALIDATION: Ensure resulting state is hardware-consistent

### REQUIRED OUTPUT FORMAT:
```
Read Value: 0x########
REGISTER_NAME_1=0x########
REGISTER_NAME_2=0x########
REGISTER_NAME_3=0x########
[... minimum {self.inference_config.min_required_registers} registers total ...]
```

"""
        else:
            return f"""### REASONING METHODOLOGY:
1. TARGET ANALYSIS: Identify target register and analyze write value
2. DIRECT EFFECTS: Apply register-specific write behavior and value updates
3. SIDE EFFECTS: Determine which other registers are affected (status, interrupts, etc.)
4. STATE CONSISTENCY: Ensure all register states remain logically consistent
5. RETURN VALUE: Calculate write operation return value (if applicable)

### REQUIRED OUTPUT FORMAT:
```
Write Return: 0x######## (if applicable)
REGISTER_NAME_1=0x########
REGISTER_NAME_2=0x########
REGISTER_NAME_3=0x########
[... minimum {self.inference_config.min_required_registers} registers total ...]
```

"""
    
    def _populate_sample_data(self, prompt: str, sample: Dict) -> str:
        """填充样本数据到提示词"""
        input_data = sample['input']
        
        # 当前寄存器状态
        regs = input_data.get('current_registers', input_data.get('previous_registers', {}))
        reg_list = []
        
        for reg_name, value in regs.items():
            reg_def = self.peripheral_config.register_definitions.get(reg_name.upper(), {})
            desc = reg_def.get('description', 'Unknown Register')
            reg_type = reg_def.get('type', 'unknown')
            
            analysis = ""
            if reg_name.upper() in self.bit_field_analysis and value in self.bit_field_analysis[reg_name.upper()]:
                analysis = f" ({self.bit_field_analysis[reg_name.upper()][value]})"
            
            reg_list.append(f"{reg_name.upper()}={value} [{reg_type}] - {desc}{analysis}")
        
        prompt += f"### CURRENT REGISTER STATE:\n"
        prompt += "\n".join(reg_list)
        prompt += f"\n\n### TARGET OPERATION DETAILS:\n"
        prompt += f"Operation Type: {input_data['operation_type']}\n"
        prompt += f"Target Offset: {input_data['offset']}\n"
        prompt += f"Access Size: {input_data['access_size']} bytes\n"
        
        if input_data['operation_type'] == 'WRITE':
            prompt += f"Write Value: {input_data['write_value']}\n"
        
        prompt += "\n### YOUR DETAILED PREDICTION:\n"
        prompt += "Apply the reasoning methodology above and provide your prediction following the exact output format.\n\n"
        
        return prompt

class EnsembleInferenceEngine:
    """集成推理引擎"""
    
    def __init__(self, model, tokenizer, config: InferenceConfig, peripheral_config: PeripheralConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.peripheral_config = peripheral_config or PeripheralConfig()
        self.prompt_engine = UniversalPromptEngine(self.peripheral_config, config)
        self.parser = RegisterParser(self.peripheral_config)
    
    def inference(self, sample: Dict, operation_type: str) -> Dict[str, Any]:
        """统一推理接口，根据配置选择单次或集成推理"""
        if self.config.enable_ensemble:
            return self.ensemble_inference(sample, operation_type)
        else:
            return self.single_inference_with_metadata(sample, operation_type)
    
    def single_inference_with_metadata(self, sample: Dict, operation_type: str) -> Dict[str, Any]:
        """单次推理（带元数据）"""
        result = self._single_inference(sample, operation_type)
        if result:
            return {
                'registers': result.get('registers', {}),
                'read_value': result.get('read_value'),
                'confidence': 1.0,  # 单次推理设置固定置信度
                'ensemble_size': 1,
                'raw_results': [result]
            }
        else:
            return {"error": "推理失败"}
    
    def ensemble_inference(self, sample: Dict, operation_type: str) -> Dict[str, Any]:
        """集成推理"""
        results = []
        
        # 生成多个不同的推理结果
        for i in range(self.config.ensemble_size):
            try:
                # 轻微调整推理参数
                temp = self.config.temperature + (i * 0.02 - 0.04)
                temp = max(0.05, min(1.0, temp))
                
                result = self._single_inference(sample, operation_type, temperature=temp)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"推理 {i+1} 失败: {e}")
        
        if not results:
            return {"error": "所有推理都失败了"}
        
        # 集成决策
        return self._ensemble_decision(results, sample, operation_type)
    
    def _single_inference(self, sample: Dict, operation_type: str, temperature: float = None) -> Optional[Dict]:
        """单次推理"""
        if temperature is None:
            temperature = self.config.temperature
        
        # 创建提示词
        prompt = self.prompt_engine.create_enhanced_prompt(sample, operation_type)
        
        # 编码
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        ).to(self.model.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析结果
        parsed_result = self._parse_response(response, operation_type)
        if parsed_result:
            parsed_result['raw_response'] = response
            parsed_result['temperature'] = temperature
        
        return parsed_result
    
    def _parse_response(self, response: str, operation_type: str) -> Optional[Dict]:
        """解析模型响应"""
        try:
            result = {}
            
            # 解析寄存器
            registers = self.parser.parse_registers(response)
            if registers:
                result['registers'] = registers
            
            # 如果是读操作，解析读取值
            if operation_type == "READ":
                read_value = self.parser.parse_read_value(response)
                if read_value:
                    result['read_value'] = read_value
            
            # 如果是写操作，解析写返回值
            if operation_type == "WRITE":
                write_return = self.parser.parse_write_return(response)
                if write_return:
                    result['write_return'] = write_return
            
            return result if result else None
            
        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return None
    
    def _ensemble_decision(self, results: List[Dict], sample: Dict, operation_type: str) -> Dict[str, Any]:
        """集成决策"""
        if not results:
            return {"error": "没有有效结果"}
        
        # 统计投票
        if operation_type == "READ":
            read_values = [r.get('read_value') for r in results if r.get('read_value')]
            if read_values:
                # 选择最频繁的读取值
                from collections import Counter
                read_counter = Counter(read_values)
                final_read_value = read_counter.most_common(1)[0][0]
                confidence = read_counter[final_read_value] / len(read_values)
            else:
                final_read_value = None
                confidence = 0.0
        else:
            final_read_value = None
            confidence = 1.0
        
        # 寄存器集成
        register_votes = defaultdict(lambda: defaultdict(int))
        for result in results:
            if 'registers' in result:
                for reg_name, value in result['registers'].items():
                    register_votes[reg_name][value] += 1
        
        # 选择每个寄存器的最频繁值
        final_registers = {}
        for reg_name, value_counts in register_votes.items():
            if value_counts:
                most_common_value = max(value_counts.items(), key=lambda x: x[1])[0]
                final_registers[reg_name] = most_common_value
        
        ensemble_result = {
            'registers': final_registers,
            'confidence': confidence,
            'ensemble_size': len(results),
            'raw_results': results
        }
        
        if final_read_value:
            ensemble_result['read_value'] = final_read_value
        
        return ensemble_result

class PeripheralModelEvaluator:
    """通用外设模型评估器"""
    
    def __init__(self, config: InferenceConfig, peripheral_config: PeripheralConfig = None):
        self.config = config
        self.peripheral_config = peripheral_config or PeripheralConfig()
        self.setup_model()
        self.inference_engine = EnsembleInferenceEngine(
            self.model, self.tokenizer, config, self.peripheral_config
        )
    
    def setup_model(self):
        """设置模型"""
        logger.info("加载模型...")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True  # Qwen模型需要这个参数
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载微调模型（如果可用）
        if self.config.use_finetuned and os.path.exists(self.config.finetuned_model_path):
            logger.info("加载微调模型...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, self.config.finetuned_model_path)
            self.model = self.model.merge_and_unload()
        else:
            logger.info(f"加载基础模型: {self.config.base_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True  # Qwen模型需要这个参数
            )
        
        logger.info("模型加载完成")
    
    def load_test_data(self) -> Tuple[List[Dict], List[Dict]]:
        """加载测试数据"""
        read_samples = []
        write_samples = []
        
        # 加载读取样本
        read_dir = os.path.join(self.config.test_data_dir, "read_samples")
        if os.path.exists(read_dir):
            files = sorted([f for f in os.listdir(read_dir) if f.endswith('.jsonl')])
            # 选择最后几个文件作为测试数据（假设是最新的）
            test_files = files[-self.config.test_files_per_type:]
            
            for file_name in test_files:
                with open(os.path.join(read_dir, file_name), 'r') as f:
                    count = 0
                    for line in f:
                        if count >= self.config.max_test_samples_per_type // self.config.test_files_per_type:
                            break
                        sample = json.loads(line.strip())
                        read_samples.append(sample)
                        count += 1
        
        # 加载写入样本
        write_dir = os.path.join(self.config.test_data_dir, "write_samples")
        if os.path.exists(write_dir):
            files = sorted([f for f in os.listdir(write_dir) if f.endswith('.jsonl')])
            # 选择最后几个文件作为测试数据
            test_files = files[-self.config.test_files_per_type:]
            
            for file_name in test_files:
                with open(os.path.join(write_dir, file_name), 'r') as f:
                    count = 0
                    for line in f:
                        if count >= self.config.max_test_samples_per_type // self.config.test_files_per_type:
                            break
                        sample = json.loads(line.strip())
                        write_samples.append(sample)
                        count += 1
        
        logger.info(f"加载测试数据: {len(read_samples)} 读取样本, {len(write_samples)} 写入样本")
        return read_samples, write_samples
    
    def evaluate_single_sample(self, sample: Dict, operation_type: str) -> Dict[str, Any]:
        """评估单个样本"""
        start_time = time.time()
        
        # 模型推理
        prediction = self.inference_engine.inference(sample, operation_type)
        
        inference_time = time.time() - start_time
        
        # 计算准确性
        ground_truth = sample['output']
        accuracy_metrics = self._calculate_comprehensive_accuracy(prediction, ground_truth, operation_type)
        
        return {
            'prediction': prediction,
            'ground_truth': ground_truth,
            'accuracy_metrics': accuracy_metrics,
            'inference_time': inference_time,
            'sample_metadata': sample.get('metadata', {}),
            'inference_mode': 'ensemble' if self.config.enable_ensemble else 'single',
            'constraint_violations': self._check_constraint_violations(prediction, operation_type)
        }
    
    def _calculate_comprehensive_accuracy(self, prediction: Dict, ground_truth: Dict, operation_type: str) -> Dict[str, float]:
        """
        计算全面的准确性指标
        
        ### 评估指标的数学逻辑和合理性解释：
        
        1. **寄存器准确率 (Register Accuracy)**:
           - 公式: correct_registers / total_registers
           - 逻辑: 衡量模型预测寄存器值的精确匹配程度
           - 合理性: 寄存器值的准确性是外设模拟的核心，错误的寄存器状态会导致整个系统行为异常
           
        2. **关键寄存器准确率 (Critical Register Accuracy)**:
           - 公式: correct_critical_registers / total_critical_registers  
           - 逻辑: 专门衡量关键寄存器（如控制、状态、数据寄存器）的准确性
           - 合理性: 关键寄存器对系统功能影响更大，应给予更高权重
           
        3. **读取值准确率 (Read Value Accuracy)** (仅读操作):
           - 公式: 1.0 if predicted_read_value == true_read_value else 0.0
           - 逻辑: 二元分类，要么完全正确要么错误
           - 合理性: 读取值是操作的直接结果，必须完全准确
           
        4. **写返回值准确率 (Write Return Accuracy)** (仅写操作):
           - 公式: 1.0 if predicted_write_return == true_write_return else 0.0  
           - 逻辑: 写操作的返回值准确性
           - 合理性: 某些写操作有返回值，反映操作是否成功
           
        5. **约束满足率 (Constraint Satisfaction Rate)**:
           - 公式: satisfied_constraints / total_constraints
           - 逻辑: 衡量输出是否满足格式和数值约束
           - 合理性: 确保模型输出符合硬件规范，避免无效预测
           
        6. **整体准确率 (Overall Accuracy)**:
           - 读操作: 0.4 * register_acc + 0.3 * critical_register_acc + 0.2 * read_value_acc + 0.1 * constraint_satisfaction
           - 写操作: 0.5 * register_acc + 0.3 * critical_register_acc + 0.1 * write_return_acc + 0.1 * constraint_satisfaction
           - 逻辑: 加权组合各个指标，权重反映重要性
           - 合理性: 
             * 寄存器准确性权重最高，因为它是状态预测的核心
             * 读取值在读操作中权重较高，因为它是操作的主要结果
             * 关键寄存器单独考虑，确保重要状态不被忽略
             * 约束满足率确保输出的基本有效性
        """
        metrics = {}
        
        # 1. 基础寄存器准确性
        register_metrics = self._calculate_register_accuracy(prediction, ground_truth)
        metrics.update(register_metrics)
        
        # 2. 关键寄存器准确性
        critical_metrics = self._calculate_critical_register_accuracy(prediction, ground_truth)
        metrics.update(critical_metrics)
        
        # 3. 操作特定准确性
        if operation_type == "READ":
            read_metrics = self._calculate_read_accuracy(prediction, ground_truth)
            metrics.update(read_metrics)
        elif operation_type == "WRITE":
            write_metrics = self._calculate_write_accuracy(prediction, ground_truth)
            metrics.update(write_metrics)
        
        # 4. 约束满足率
        constraint_metrics = self._calculate_constraint_satisfaction(prediction, operation_type)
        metrics.update(constraint_metrics)
        
        # 5. 计算整体准确率
        metrics['overall_accuracy'] = self._calculate_weighted_overall_accuracy(metrics, operation_type)
        
        return metrics
    
    def _calculate_register_accuracy(self, prediction: Dict, ground_truth: Dict) -> Dict[str, float]:
        """计算基础寄存器准确性"""
        if 'registers' not in prediction or 'resulting_registers' not in ground_truth:
            return {'register_accuracy': 0.0, 'register_count': 0}
        
        pred_regs = prediction['registers']
        true_regs = ground_truth['resulting_registers']
        
        correct_regs = 0
        total_regs = len(true_regs)
        
        for reg_name, true_value in true_regs.items():
            pred_value = pred_regs.get(reg_name, "")
            if pred_value.lower() == true_value.lower():
                correct_regs += 1
        
        return {
            'register_accuracy': correct_regs / total_regs if total_regs > 0 else 0.0,
            'register_count': total_regs,
            'correct_register_count': correct_regs
        }
    
    def _calculate_critical_register_accuracy(self, prediction: Dict, ground_truth: Dict) -> Dict[str, float]:
        """计算关键寄存器准确性"""
        if 'registers' not in prediction or 'resulting_registers' not in ground_truth:
            return {'critical_register_accuracy': 0.0, 'critical_register_count': 0}
        
        pred_regs = prediction['registers']
        true_regs = ground_truth['resulting_registers']
        critical_regs = self.peripheral_config.critical_registers
        
        correct_critical = 0
        total_critical = 0
        
        for reg_name in critical_regs:
            if reg_name.lower() in true_regs:
                total_critical += 1
                true_value = true_regs[reg_name.lower()]
                pred_value = pred_regs.get(reg_name.lower(), "")
                if pred_value.lower() == true_value.lower():
                    correct_critical += 1
        
        return {
            'critical_register_accuracy': correct_critical / total_critical if total_critical > 0 else 0.0,
            'critical_register_count': total_critical,
            'correct_critical_count': correct_critical
        }
    
    def _calculate_read_accuracy(self, prediction: Dict, ground_truth: Dict) -> Dict[str, float]:
        """计算读操作准确性"""
        if 'read_value' in prediction and 'read_value' in ground_truth:
            pred_val = prediction['read_value'].lower()
            true_val = ground_truth['read_value'].lower()
            return {'read_value_accuracy': 1.0 if pred_val == true_val else 0.0}
        else:
            return {'read_value_accuracy': 0.0}
    
    def _calculate_write_accuracy(self, prediction: Dict, ground_truth: Dict) -> Dict[str, float]:
        """计算写操作准确性"""
        metrics = {}
        
        # 写返回值准确性（如果存在）
        if 'write_return' in prediction and 'write_return' in ground_truth:
            pred_val = prediction['write_return'].lower()
            true_val = ground_truth['write_return'].lower()
            metrics['write_return_accuracy'] = 1.0 if pred_val == true_val else 0.0
        else:
            metrics['write_return_accuracy'] = 0.0
        
        return metrics
    
    def _calculate_constraint_satisfaction(self, prediction: Dict, operation_type: str) -> Dict[str, float]:
        """计算约束满足率"""
        violations = self._check_constraint_violations(prediction, operation_type)
        total_constraints = len(violations)
        satisfied_constraints = sum(1 for v in violations.values() if not v)
        
        return {
            'constraint_satisfaction_rate': satisfied_constraints / total_constraints if total_constraints > 0 else 1.0,
            'total_constraints': total_constraints,
            'satisfied_constraints': satisfied_constraints
        }
    
    def _check_constraint_violations(self, prediction: Dict, operation_type: str) -> Dict[str, bool]:
        """检查约束违规"""
        violations = {}
        
        if 'registers' in prediction:
            pred_regs = prediction['registers']
            
            # 检查寄存器数量约束
            violations['insufficient_registers'] = len(pred_regs) < self.config.min_required_registers
            
            # 检查寄存器值范围
            for reg_name, value in pred_regs.items():
                try:
                    val_int = int(value, 16)
                    min_val = int(self.config.min_register_value, 16)
                    max_val = int(self.config.max_register_value, 16)
                    violations[f'{reg_name}_out_of_range'] = not (min_val <= val_int <= max_val)
                except ValueError:
                    violations[f'{reg_name}_invalid_format'] = True
            
            # 检查关键寄存器是否存在
            for critical_reg in self.peripheral_config.critical_registers:
                violations[f'missing_{critical_reg.lower()}'] = critical_reg.lower() not in pred_regs
        
        # 检查操作特定约束
        if operation_type == "READ" and self.config.require_return_value:
            violations['missing_read_value'] = 'read_value' not in prediction
        
        return violations
    
    def _calculate_weighted_overall_accuracy(self, metrics: Dict[str, float], operation_type: str) -> float:
        """计算加权整体准确率"""
        reg_acc = metrics.get('register_accuracy', 0.0)
        critical_acc = metrics.get('critical_register_accuracy', 0.0)
        constraint_sat = metrics.get('constraint_satisfaction_rate', 0.0)
        
        if operation_type == "READ":
            read_acc = metrics.get('read_value_accuracy', 0.0)
            # 读操作权重分配：寄存器40%，关键寄存器30%，读取值20%，约束10%
            return 0.4 * reg_acc + 0.3 * critical_acc + 0.2 * read_acc + 0.1 * constraint_sat
        else:
            write_acc = metrics.get('write_return_accuracy', 0.0)
            # 写操作权重分配：寄存器50%，关键寄存器30%，写返回10%，约束10%
            return 0.5 * reg_acc + 0.3 * critical_acc + 0.1 * write_acc + 0.1 * constraint_sat

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """运行全面评估"""
        logger.info("开始全面模型评估...")
        
        # 加载测试数据
        read_samples, write_samples = self.load_test_data()
        
        # 确保使用单次推理
        self.config.enable_ensemble = False
        
        read_results = []
        write_results = []
        
        # 评估读取操作
        for i, sample in enumerate(read_samples):
            logger.info(f"评估读取样本 {i+1}/{len(read_samples)}")
            try:
                result = self.evaluate_single_sample(sample, "READ")
                read_results.append(result)
            except Exception as e:
                logger.error(f"评估读取样本 {i+1} 失败: {e}")
        
        # 评估写入操作
        for i, sample in enumerate(write_samples):
            logger.info(f"评估写入样本 {i+1}/{len(write_samples)}")
            try:
                result = self.evaluate_single_sample(sample, "WRITE")
                write_results.append(result)
            except Exception as e:
                logger.error(f"评估写入样本 {i+1} 失败: {e}")
        
        # 创建评估摘要
        evaluation_summary = self._create_comprehensive_evaluation_summary(read_results, write_results)
        
        # 保存结果
        self._save_comprehensive_evaluation_results(read_results, write_results, evaluation_summary)
        
        return evaluation_summary
    
    def _create_comprehensive_evaluation_summary(self, read_results: List[Dict], write_results: List[Dict]) -> Dict[str, Any]:
        """创建全面评估摘要"""
        def summarize_comprehensive_results(results, operation_type):
            if not results:
                return {'total_samples': 0}
            
            # 提取所有指标
            overall_acc = []
            register_acc = []
            critical_register_acc = []
            constraint_satisfaction = []
            inference_times = []
            
            # 操作特定指标
            operation_specific_acc = []
            
            # 约束违规统计
            violation_counts = defaultdict(int)
            
            for r in results:
                accuracy_metrics = r.get('accuracy_metrics', {})
                overall_acc.append(accuracy_metrics.get('overall_accuracy', 0.0))
                register_acc.append(accuracy_metrics.get('register_accuracy', 0.0))
                critical_register_acc.append(accuracy_metrics.get('critical_register_accuracy', 0.0))
                constraint_satisfaction.append(accuracy_metrics.get('constraint_satisfaction_rate', 0.0))
                inference_times.append(r.get('inference_time', 0.0))
                
                # 操作特定准确性
                if operation_type == "READ":
                    operation_specific_acc.append(accuracy_metrics.get('read_value_accuracy', 0.0))
                else:
                    operation_specific_acc.append(accuracy_metrics.get('write_return_accuracy', 0.0))
                
                # 统计约束违规
                violations = r.get('constraint_violations', {})
                for violation, is_violated in violations.items():
                    if is_violated:
                        violation_counts[violation] += 1
            
            summary = {
                'operation_type': operation_type,
                'total_samples': len(results),
                'overall_accuracy': {
                    'mean': np.mean(overall_acc),
                    'std': np.std(overall_acc),
                    'min': np.min(overall_acc),
                    'max': np.max(overall_acc)
                },
                'register_accuracy': {
                    'mean': np.mean(register_acc),
                    'std': np.std(register_acc),
                    'min': np.min(register_acc),
                    'max': np.max(register_acc)
                },
                'critical_register_accuracy': {
                    'mean': np.mean(critical_register_acc),
                    'std': np.std(critical_register_acc),
                    'min': np.min(critical_register_acc),
                    'max': np.max(critical_register_acc)
                },
                'constraint_satisfaction_rate': {
                    'mean': np.mean(constraint_satisfaction),
                    'std': np.std(constraint_satisfaction),
                    'min': np.min(constraint_satisfaction),
                    'max': np.max(constraint_satisfaction)
                },
                'inference_time': {
                    'mean': np.mean(inference_times),
                    'std': np.std(inference_times),
                    'min': np.min(inference_times),
                    'max': np.max(inference_times)
                },
                'constraint_violations': dict(violation_counts)
            }
            
            # 添加操作特定指标
            if operation_type == "READ":
                summary['read_value_accuracy'] = {
                    'mean': np.mean(operation_specific_acc),
                    'std': np.std(operation_specific_acc),
                    'min': np.min(operation_specific_acc),
                    'max': np.max(operation_specific_acc)
                }
            else:
                summary['write_return_accuracy'] = {
                    'mean': np.mean(operation_specific_acc),
                    'std': np.std(operation_specific_acc),
                    'min': np.min(operation_specific_acc),
                    'max': np.max(operation_specific_acc)
                }
            
            return summary
        
        return {
            'read_operations': summarize_comprehensive_results(read_results, 'READ'),
            'write_operations': summarize_comprehensive_results(write_results, 'WRITE'),
            'model_info': {
                'base_model': self.config.base_model_name,
                'use_finetuned': self.config.use_finetuned,
                'inference_mode': 'ensemble' if self.config.enable_ensemble else 'single'
            },
            'peripheral_info': {
                'type': self.peripheral_config.peripheral_type,
                'name': self.peripheral_config.peripheral_name,
                'critical_registers': self.peripheral_config.critical_registers
            },
            'evaluation_constraints': {
                'min_required_registers': self.config.min_required_registers,
                'register_value_range': f"{self.config.min_register_value} - {self.config.max_register_value}",
                'enforce_format_constraints': self.config.enforce_format_constraints
            }
        }
    
    def _save_comprehensive_evaluation_results(self, read_results: List[Dict], write_results: List[Dict], summary: Dict):
        """保存全面评估结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存详细结果
        results_data = {
            'read_results': read_results,
            'write_results': write_results,
            'evaluation_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'peripheral_config': {
                    'type': self.peripheral_config.peripheral_type,
                    'name': self.peripheral_config.peripheral_name,
                    'critical_registers': self.peripheral_config.critical_registers
                },
                'inference_config': {
                    'min_required_registers': self.config.min_required_registers,
                    'value_range': f"{self.config.min_register_value} - {self.config.max_register_value}",
                    'enforce_constraints': self.config.enforce_format_constraints
                }
            }
        }
        
        with open(os.path.join(self.config.output_dir, "comprehensive_evaluation_results.json"), 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # 保存摘要
        with open(os.path.join(self.config.output_dir, "comprehensive_evaluation_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # 保存约束违规详细分析
        violation_analysis = self._analyze_constraint_violations(read_results, write_results)
        with open(os.path.join(self.config.output_dir, "constraint_violation_analysis.json"), 'w') as f:
            json.dump(violation_analysis, f, indent=2, default=str)
        
        logger.info(f"全面评估结果已保存到: {self.config.output_dir}")
    
    def _analyze_constraint_violations(self, read_results: List[Dict], write_results: List[Dict]) -> Dict[str, Any]:
        """分析约束违规模式"""
        all_results = read_results + write_results
        violation_patterns = defaultdict(lambda: {'count': 0, 'samples': []})
        
        for i, result in enumerate(all_results):
            violations = result.get('constraint_violations', {})
            sample_violations = [v for v, is_violated in violations.items() if is_violated]
            
            for violation in sample_violations:
                violation_patterns[violation]['count'] += 1
                violation_patterns[violation]['samples'].append(i)
        
        # 计算违规率
        total_samples = len(all_results)
        violation_analysis = {
            'total_samples': total_samples,
            'violation_summary': {},
            'recommendations': []
        }
        
        for violation, data in violation_patterns.items():
            violation_rate = data['count'] / total_samples
            violation_analysis['violation_summary'][violation] = {
                'count': data['count'],
                'rate': violation_rate,
                'severity': 'high' if violation_rate > 0.3 else 'medium' if violation_rate > 0.1 else 'low'
            }
            
            # 生成建议
            if violation_rate > 0.3:
                if 'insufficient_registers' in violation:
                    violation_analysis['recommendations'].append(
                        f"高频违规 ({violation_rate:.1%}): 寄存器数量不足，建议在训练时强调输出完整的寄存器状态"
                    )
                elif 'out_of_range' in violation:
                    violation_analysis['recommendations'].append(
                        f"高频违规 ({violation_rate:.1%}): 寄存器值超出范围，建议添加数值约束训练"
                    )
                elif 'missing_' in violation:
                    violation_analysis['recommendations'].append(
                        f"高频违规 ({violation_rate:.1%}): 缺少关键寄存器，建议重点训练关键寄存器预测"
                    )
        
        return violation_analysis

def main():
    """主函数"""
    config = InferenceConfig()
    peripheral_config = PeripheralConfig()
    
    # 创建通用外设评估器
    evaluator = PeripheralModelEvaluator(config, peripheral_config)
    
    # 运行评估
    summary = evaluator.run_comprehensive_evaluation()
    
    # 打印详细结果摘要
    print("\n" + "="*100)
    print(f"通用外设模型评估结果 - {peripheral_config.peripheral_name} {peripheral_config.peripheral_type}")
    print("="*100)
    
    # 模型和外设信息
    model_info = summary.get('model_info', {})
    print(f"\n【模型配置】")
    print(f"基础模型: {model_info.get('base_model', 'Unknown')}")
    print(f"使用微调: {model_info.get('use_finetuned', False)}")
    print(f"推理模式: {model_info.get('inference_mode', 'single')}")
    
    print(f"\n【外设配置】")
    print(f"外设类型: {peripheral_config.peripheral_type}")
    print(f"外设名称: {peripheral_config.peripheral_name}")
    print(f"基地址: {peripheral_config.base_address}")
    print(f"关键寄存器: {', '.join(peripheral_config.critical_registers)}")
    
    print(f"\n【推理约束】")
    print(f"最少寄存器数: {config.min_required_registers}")
    print(f"寄存器值范围: {config.min_register_value} - {config.max_register_value}")
    print(f"强制格式约束: {config.enforce_format_constraints}")
    
    # 读取操作详细结果
    read_op = summary.get('read_operations', {})
    if read_op.get('total_samples', 0) > 0:
        print(f"\n【读取操作详细分析】({read_op.get('total_samples', 0)} 样本)")
        print(f"  整体准确率: {read_op.get('overall_accuracy', {}).get('mean', 0):.3f} ± {read_op.get('overall_accuracy', {}).get('std', 0):.3f}")
        print(f"  基础寄存器准确率: {read_op.get('register_accuracy', {}).get('mean', 0):.3f} ± {read_op.get('register_accuracy', {}).get('std', 0):.3f}")
        print(f"  关键寄存器准确率: {read_op.get('critical_register_accuracy', {}).get('mean', 0):.3f} ± {read_op.get('critical_register_accuracy', {}).get('std', 0):.3f}")
        print(f"  读取值准确率: {read_op.get('read_value_accuracy', {}).get('mean', 0):.3f} ± {read_op.get('read_value_accuracy', {}).get('std', 0):.3f}")
        print(f"  约束满足率: {read_op.get('constraint_satisfaction_rate', {}).get('mean', 0):.3f} ± {read_op.get('constraint_satisfaction_rate', {}).get('std', 0):.3f}")
        print(f"  平均推理时间: {read_op.get('inference_time', {}).get('mean', 0):.3f}s")
        
        # 约束违规分析
        if 'constraint_violations' in read_op:
            violations = read_op['constraint_violations']
            print(f"  常见约束违规:")
            for violation, count in violations.items():
                if count > 0:
                    print(f"    - {violation}: {count} 次")
    
    # 写入操作详细结果
    write_op = summary.get('write_operations', {})
    if write_op.get('total_samples', 0) > 0:
        print(f"\n【写入操作详细分析】({write_op.get('total_samples', 0)} 样本)")
        print(f"  整体准确率: {write_op.get('overall_accuracy', {}).get('mean', 0):.3f} ± {write_op.get('overall_accuracy', {}).get('std', 0):.3f}")
        print(f"  基础寄存器准确率: {write_op.get('register_accuracy', {}).get('mean', 0):.3f} ± {write_op.get('register_accuracy', {}).get('std', 0):.3f}")
        print(f"  关键寄存器准确率: {write_op.get('critical_register_accuracy', {}).get('mean', 0):.3f} ± {write_op.get('critical_register_accuracy', {}).get('std', 0):.3f}")
        print(f"  写返回值准确率: {write_op.get('write_return_accuracy', {}).get('mean', 0):.3f} ± {write_op.get('write_return_accuracy', {}).get('std', 0):.3f}")
        print(f"  约束满足率: {write_op.get('constraint_satisfaction_rate', {}).get('mean', 0):.3f} ± {write_op.get('constraint_satisfaction_rate', {}).get('std', 0):.3f}")
        print(f"  平均推理时间: {write_op.get('inference_time', {}).get('mean', 0):.3f}s")
        
        # 约束违规分析
        if 'constraint_violations' in write_op:
            violations = write_op['constraint_violations']
            print(f"  常见约束违规:")
            for violation, count in violations.items():
                if count > 0:
                    print(f"    - {violation}: {count} 次")
    
    # 整体统计和评估分析
    all_samples = read_op.get('total_samples', 0) + write_op.get('total_samples', 0)
    if all_samples > 0:
        overall_acc = (
            read_op.get('overall_accuracy', {}).get('mean', 0) * read_op.get('total_samples', 0) +
            write_op.get('overall_accuracy', {}).get('mean', 0) * write_op.get('total_samples', 0)
        ) / all_samples
        
        overall_constraint_sat = (
            read_op.get('constraint_satisfaction_rate', {}).get('mean', 0) * read_op.get('total_samples', 0) +
            write_op.get('constraint_satisfaction_rate', {}).get('mean', 0) * write_op.get('total_samples', 0)
        ) / all_samples
        
        print(f"\n【整体评估分析】")
        print(f"总样本数: {all_samples}")
        print(f"综合准确率: {overall_acc:.3f}")
        print(f"综合约束满足率: {overall_constraint_sat:.3f}")
        
        # 性能等级评估
        if overall_acc >= 0.9:
            performance_level = "优秀 (Excellent)"
        elif overall_acc >= 0.8:
            performance_level = "良好 (Good)"
        elif overall_acc >= 0.7:
            performance_level = "可接受 (Acceptable)"
        elif overall_acc >= 0.6:
            performance_level = "需要改进 (Needs Improvement)"
        else:
            performance_level = "较差 (Poor)"
        
        print(f"性能等级: {performance_level}")
        
        # 推荐建议
        print(f"\n【推荐建议】")
        if overall_acc < 0.8:
            print("- 建议增加训练数据或调整模型参数")
        if overall_constraint_sat < 0.9:
            print("- 建议加强输出格式约束的训练")
        if read_op.get('read_value_accuracy', {}).get('mean', 0) < 0.8:
            print("- 建议重点优化读取值预测的准确性")
        if write_op.get('critical_register_accuracy', {}).get('mean', 0) < 0.8:
            print("- 建议加强关键寄存器状态变化的训练")
    
    print(f"\n详细结果已保存到: {config.output_dir}")
    print("="*100)
    
    print(f"\n【评估指标说明】")
    print("1. 整体准确率: 综合考虑寄存器、关键寄存器、操作值和约束的加权准确率")
    print("2. 基础寄存器准确率: 所有预测寄存器值与真实值的匹配率")
    print("3. 关键寄存器准确率: 重要寄存器（控制、状态、数据）的预测准确率")
    print("4. 读取值/写返回值准确率: 操作直接结果的预测准确率")
    print("5. 约束满足率: 输出格式、数值范围、数量要求等约束的满足程度")
    print("6. 权重设计: 读操作(寄存器40%+关键30%+读值20%+约束10%), 写操作(寄存器50%+关键30%+写返回10%+约束10%)")

if __name__ == "__main__":
    main() 
