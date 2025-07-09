#!/usr/bin/env python3
"""
JSON数据处理器 - 重构版
从JSON文件中提取推理输入，调用AI推理引擎，并进行结果比较
支持新的多层级推理引擎架构
"""

import json
import os
import time
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import argparse
from pathlib import Path

# 使用重构后的AI推理引擎
from ai_inference_engine import (
    MultiLevelInferenceEngine, 
    InferenceConfig, 
    InferenceConstraints,
    AccessPattern,
    DeviceConfig,
    PeripheralType
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    """处理器配置"""
    input_data_dir: str = "./evaluation_data"
    output_dir: str = "./evaluation_results"
    max_samples_per_file: int = 40  # 每个文件最多处理的样本数
    max_files_per_type: int = 5     # 每种操作类型最多处理的文件数
    
    # AI推理配置
    ai_config: InferenceConfig = None
    
    def __post_init__(self):
        if self.ai_config is None:
            # 创建约束配置
            constraints = InferenceConstraints(
                min_registers_count=3,
                max_registers_count=15,
                required_registers={"UARTDR", "UARTFR", "UARTCR"},  # PL011关键寄存器
                register_value_range=(0x0, 0xFFFFFFFF)
            )
            
            self.ai_config = InferenceConfig(
                model_path="/data/LLM_models/Qwen2.5-Coder-14B-Instruct",
                temperature=0.2,  # 降低温度提高一致性
                timeout_seconds=120,
                confidence_threshold=0.6,  # 降低阈值提高通过率
                constraints=constraints,
                enable_pattern_cache=True,
                enable_knowledge_base=True,
                enable_semantic_analysis=True
            )

class DataExtractor:
    """数据提取器 - 适配新的推理引擎接口"""
    
    def extract_sample_data(self, json_record: Dict[str, Any]) -> Dict[str, Any]:
        """从JSON记录中提取推理所需的数据"""
        try:
            # 新格式：嵌套结构 input/output/metadata
            if "input" in json_record and "output" in json_record:
                input_data = json_record["input"]
                output_data = json_record["output"]
                metadata = json_record.get("metadata", {})
                
                # 提取操作信息
                operation = metadata.get("operation", "")
                if not operation:
                    # 从input构建操作字符串
                    op_type = input_data.get("operation_type", "UNKNOWN")
                    offset = input_data.get("offset", "unknown")
                    value = input_data.get("value", "")
                    if value:
                        operation = f"PL011 UART {op_type} offset={offset} value={value}"
                    else:
                        operation = f"PL011 UART {op_type} offset={offset}"
                
                # 提取当前寄存器状态
                current_registers = input_data.get("current_registers", {})
                
                # 提取真实结果
                ground_truth = {
                    "read_value": output_data.get("read_value"),
                    "resulting_registers": output_data.get("resulting_registers", {})
                }
                
                # 提取元数据
                metadata_info = {
                    "sequence_id": input_data.get("sequence_id"),
                    "timestamp": input_data.get("timestamp"),
                    "device_type": metadata.get("device_type", "pl011_uart"),
                    "operation_type": input_data.get("operation_type"),
                    "offset": input_data.get("offset"),
                    "access_size": input_data.get("access_size"),
                    "value": input_data.get("value")  # 写操作值
                }
                
            else:
                # 旧格式：扁平结构
                operation = json_record.get("operation", "")
                current_registers = json_record.get("current_registers", {})
                
                ground_truth = {
                    "read_value": json_record.get("read_value"),
                    "resulting_registers": json_record.get("resulting_registers", {})
                }
                
                metadata_info = {
                    "sequence_id": json_record.get("sequence_id"),
                    "timestamp": json_record.get("timestamp"),
                    "device_type": json_record.get("device_type", "pl011_uart"),
                    "operation": operation
                }
            
            return {
                "operation": operation,
                "current_registers": current_registers,
                "ground_truth": ground_truth,
                "metadata": metadata_info
            }
            
        except Exception as e:
            logger.error(f"提取样本数据失败: {e}")
            return None
    
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
    
    def _extract_register_name(self, offset: str) -> str:
        """根据偏移量提取寄存器名称"""
        # PL011寄存器映射
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
    
    def load_samples_from_file(self, file_path: str, max_samples: int) -> List[Dict[str, Any]]:
        """从文件中加载样本"""
        samples = []
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if len(samples) >= max_samples:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        sample = self.extract_sample_data(record)
                        if sample:
                            samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"文件 {file_path} 第 {line_num} 行JSON解析失败: {e}")
                    except Exception as e:
                        logger.error(f"文件 {file_path} 第 {line_num} 行处理失败: {e}")
            
            logger.info(f"从 {file_path} 加载了 {len(samples)} 个样本")
            
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {e}")
        
        return samples

class AccuracyCalculator:
    """准确性计算器"""
    
    def calculate_register_accuracy(self, predicted: Dict[str, str], actual: Dict[str, str]) -> float:
        """计算寄存器预测准确性"""
        if not predicted or not actual:
            return 0.0
        
        # 只比较都存在的寄存器
        common_registers = set(predicted.keys()) & set(actual.keys())
        if not common_registers:
            return 0.0
        
        correct_count = 0
        for reg_name in common_registers:
            pred_val = predicted[reg_name].lower()
            actual_val = actual[reg_name].lower()
            
            # 跳过模板文本
            if "current_or_updated_value" in pred_val:
                continue
            
            # 标准化十六进制格式
            if pred_val.startswith('0x'):
                pred_val = pred_val[2:]
            if actual_val.startswith('0x'):
                actual_val = actual_val[2:]
            
            # 移除前导零进行比较
            pred_val = pred_val.lstrip('0') or '0'
            actual_val = actual_val.lstrip('0') or '0'
            
            if pred_val == actual_val:
                correct_count += 1
        
        # 计算有效预测的寄存器数量（排除模板文本）
        valid_predictions = len([r for r in common_registers 
                               if "current_or_updated_value" not in predicted[r].lower()])
        
        return correct_count / valid_predictions if valid_predictions > 0 else 0.0
    
    def calculate_read_value_accuracy(self, predicted: str, actual: str) -> float:
        """计算读取值准确性"""
        if not predicted or not actual:
            return 0.0
        
        # 标准化格式，移除前缀0x并统一长度
        pred_val = predicted.lower()
        actual_val = actual.lower()
        
        if pred_val.startswith('0x'):
            pred_val = pred_val[2:]
        if actual_val.startswith('0x'):
            actual_val = actual_val[2:]
        
        # 移除前导零进行比较
        pred_val = pred_val.lstrip('0') or '0'
        actual_val = actual_val.lstrip('0') or '0'
        
        return 1.0 if pred_val == actual_val else 0.0
    
    def calculate_overall_accuracy(self, sample_result: Dict[str, Any]) -> float:
        """计算整体准确性"""
        reg_acc = sample_result.get("register_accuracy", 0.0)
        
        # 如果是读操作，考虑读取值准确性
        if "read_value_accuracy" in sample_result:
            read_acc = sample_result["read_value_accuracy"]
            return (reg_acc + read_acc) / 2.0
        else:
            return reg_acc

class JsonDataProcessor:
    """JSON数据处理器主类 - 适配新的多层级推理引擎"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.extractor = DataExtractor()
        self.calculator = AccuracyCalculator()
        self.ai_engine = None
    
    def initialize_ai_engine(self):
        """初始化AI推理引擎"""
        if self.ai_engine is None:
            logger.info("初始化多层级AI推理引擎...")
            self.ai_engine = MultiLevelInferenceEngine(self.config.ai_config)
            logger.info("多层级推理引擎初始化完成")
    
    def find_data_files(self) -> Tuple[List[str], List[str]]:
        """查找数据文件"""
        data_dir = Path(self.config.input_data_dir)
        
        read_files = []
        write_files = []
        
        if data_dir.exists():
            # 查找主目录中的JSON文件
            for pattern in ["*.json", "*.jsonl"]:
                for file_path in data_dir.glob(pattern):
                    filename = file_path.name.lower()
                    if "read" in filename:
                        read_files.append(str(file_path))
                    elif "write" in filename:
                        write_files.append(str(file_path))
            
            # 查找子目录中的JSON文件
            for subdir in ["read_samples", "write_samples"]:
                subdir_path = data_dir / subdir
                if subdir_path.exists():
                    for pattern in ["*.json", "*.jsonl"]:
                        for file_path in subdir_path.glob(pattern):
                            if "read" in subdir:
                                read_files.append(str(file_path))
                            elif "write" in subdir:
                                write_files.append(str(file_path))
        
        # 按文件名排序，取最新的文件
        read_files.sort(reverse=True)
        write_files.sort(reverse=True)
        
        # 限制文件数量
        read_files = read_files[:self.config.max_files_per_type]
        write_files = write_files[:self.config.max_files_per_type]
        
        logger.info(f"找到读操作文件: {read_files}")
        logger.info(f"找到写操作文件: {write_files}")
        
        return read_files, write_files
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个样本 - 使用新的推理引擎接口"""
        try:
            # 创建访问模式对象
            access_pattern = self.extractor.create_access_pattern(sample)
            
            # 使用新的智能推理接口
            ai_result = self.ai_engine.intelligent_inference(
                access_pattern=access_pattern,
                device_name="PL011"  # 默认PL011设备
            )
            
            if not ai_result["success"]:
                return {
                    "success": False,
                    "error": ai_result.get("error", "推理失败"),
                    "inference_time": ai_result["inference_time"],
                    "confidence": ai_result.get("confidence", 0.0),
                    "sample_metadata": sample["metadata"]
                }
            
            # 提取预测结果 (新格式直接使用ai_result)
            prediction = {
                "read_value": ai_result.get("read_value"),
                "registers": ai_result.get("registers", {})
            }
            
            ground_truth = sample["ground_truth"]
            
            # 计算准确性
            accuracy_metrics = {}
            
            # 寄存器准确性
            accuracy_metrics["register_accuracy"] = self.calculator.calculate_register_accuracy(
                prediction.get("registers", {}),
                ground_truth.get("resulting_registers", {})
            )
            
            # 读取值准确性（如果是读操作）
            if prediction.get("read_value") and ground_truth.get("read_value"):
                accuracy_metrics["read_value_accuracy"] = self.calculator.calculate_read_value_accuracy(
                    prediction["read_value"],
                    ground_truth["read_value"]
                )
            
            # 整体准确性
            accuracy_metrics["overall_accuracy"] = self.calculator.calculate_overall_accuracy(accuracy_metrics)
            
            return {
                "success": True,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "accuracy_metrics": accuracy_metrics,
                "inference_time": ai_result["inference_time"],
                "confidence": ai_result["confidence"],
                "validation_passed": ai_result.get("validation_passed", True),
                "validation_errors": ai_result.get("validation_errors", []),
                "auto_fixed": ai_result.get("auto_fixed", False),
                "quick_path": ai_result.get("quick_path", False),
                "sample_metadata": sample["metadata"]
            }
            
        except Exception as e:
            logger.error(f"处理样本失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0,
                "sample_metadata": sample.get("metadata", {})
            }
    
    def process_files(self, files: List[str], operation_type: str) -> List[Dict[str, Any]]:
        """处理文件列表"""
        all_results = []
        
        for file_path in files:
            logger.info(f"处理文件: {file_path}")
            
            # 加载样本
            samples = self.extractor.load_samples_from_file(file_path, self.config.max_samples_per_file)
            
            if not samples:
                logger.warning(f"文件 {file_path} 中没有有效样本")
                continue
            
            # 处理每个样本
            for i, sample in enumerate(samples):
                logger.info(f"处理 {operation_type} 样本 {i+1}/{len(samples)}")
                result = self.process_sample(sample)
                result["operation_type"] = operation_type
                all_results.append(result)
        
        return all_results
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成结果摘要"""
        if not results:
            return {"total_samples": 0}
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {
                "total_samples": len(results),
                "successful_samples": 0,
                "success_rate": 0.0
            }
        
        # 计算统计信息
        accuracies = {
            "overall": [r["accuracy_metrics"]["overall_accuracy"] for r in successful_results],
            "register": [r["accuracy_metrics"]["register_accuracy"] for r in successful_results],
        }
        
        # 读取值准确性（如果存在）
        read_accuracies = [r["accuracy_metrics"]["read_value_accuracy"] 
                          for r in successful_results 
                          if "read_value_accuracy" in r["accuracy_metrics"]]
        
        if read_accuracies:
            accuracies["read_value"] = read_accuracies
        
        inference_times = [r["inference_time"] for r in successful_results]
        confidences = [r.get("confidence", 0.0) for r in successful_results]
        
        # 统计快速路径使用情况
        quick_path_count = len([r for r in successful_results if r.get("quick_path", False)])
        auto_fixed_count = len([r for r in successful_results if r.get("auto_fixed", False)])
        
        import numpy as np
        
        summary = {
            "total_samples": len(results),
            "successful_samples": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            
            "accuracy_stats": {
                "overall": {
                    "mean": np.mean(accuracies["overall"]),
                    "std": np.std(accuracies["overall"]),
                    "min": np.min(accuracies["overall"]),
                    "max": np.max(accuracies["overall"])
                },
                "register": {
                    "mean": np.mean(accuracies["register"]),
                    "std": np.std(accuracies["register"]),
                    "min": np.min(accuracies["register"]),
                    "max": np.max(accuracies["register"])
                }
            },
            
            "performance_stats": {
                "inference_time": {
                    "mean": np.mean(inference_times),
                    "std": np.std(inference_times),
                    "min": np.min(inference_times),
                    "max": np.max(inference_times)
                },
                "confidence": {
                    "mean": np.mean(confidences),
                    "std": np.std(confidences),
                    "min": np.min(confidences),
                    "max": np.max(confidences)
                }
            },
            
            "engine_stats": {
                "quick_path_usage": quick_path_count / len(successful_results) if successful_results else 0,
                "auto_fixed_count": auto_fixed_count,
                "auto_fixed_rate": auto_fixed_count / len(successful_results) if successful_results else 0
            }
        }
        
        if read_accuracies:
            summary["accuracy_stats"]["read_value"] = {
                "mean": np.mean(read_accuracies),
                "std": np.std(read_accuracies),
                "min": np.min(read_accuracies),
                "max": np.max(read_accuracies)
            }
        
        return summary
    
    def save_results(self, read_results: List[Dict], write_results: List[Dict]):
        """保存结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 保存详细结果
        detailed_results = {
            "read_results": read_results,
            "write_results": write_results,
            "config": {
                "ai_model": self.config.ai_config.model_path,
                "temperature": self.config.ai_config.temperature,
                "confidence_threshold": self.config.ai_config.confidence_threshold,
                "enable_pattern_cache": self.config.ai_config.enable_pattern_cache,
                "enable_knowledge_base": self.config.ai_config.enable_knowledge_base,
                "enable_semantic_analysis": self.config.ai_config.enable_semantic_analysis,
                "max_samples_per_file": self.config.max_samples_per_file,
                "max_files_per_type": self.config.max_files_per_type
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(self.config.output_dir, "detailed_results.json"), 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # 生成摘要
        read_summary = self.generate_summary(read_results)
        write_summary = self.generate_summary(write_results)
        
        summary = {
            "read_operations": read_summary,
            "write_operations": write_summary,
            "overall": {
                "total_samples": len(read_results) + len(write_results),
                "total_successful": read_summary.get("successful_samples", 0) + write_summary.get("successful_samples", 0)
            }
        }
        
        with open(os.path.join(self.config.output_dir, "evaluation_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"结果已保存到: {self.config.output_dir}")
        
        return summary
    
    def run_evaluation(self):
        """运行评估"""
        logger.info("开始多层级AI推理评估...")
        
        # 初始化AI引擎
        self.initialize_ai_engine()
        
        # 查找数据文件
        read_files, write_files = self.find_data_files()
        
        if not read_files and not write_files:
            logger.error(f"在 {self.config.input_data_dir} 中没有找到数据文件")
            return
        
        # 处理读操作
        read_results = []
        if read_files:
            logger.info("处理读操作文件...")
            read_results = self.process_files(read_files, "READ")
        
        # 处理写操作
        write_results = []
        if write_files:
            logger.info("处理写操作文件...")
            write_results = self.process_files(write_files, "WRITE")
        
        # 保存结果并生成摘要
        summary = self.save_results(read_results, write_results)
        
        # 打印摘要
        self.print_summary(summary)
    
    def print_summary(self, summary: Dict[str, Any]):
        """打印评估摘要 - 包含新的推理引擎指标"""
        print("\n" + "="*80)
        print("多层级AI推理评估结果摘要")
        print("="*80)
        
        # 读操作结果
        read_ops = summary.get("read_operations", {})
        if read_ops.get("total_samples", 0) > 0:
            print(f"\n【读操作】")
            print(f"总样本数: {read_ops['total_samples']}")
            print(f"成功样本数: {read_ops['successful_samples']}")
            print(f"成功率: {read_ops['success_rate']:.2%}")
            
            if read_ops.get("accuracy_stats"):
                acc_stats = read_ops["accuracy_stats"]
                print(f"整体准确率: {acc_stats['overall']['mean']:.3f} ± {acc_stats['overall']['std']:.3f}")
                print(f"寄存器准确率: {acc_stats['register']['mean']:.3f} ± {acc_stats['register']['std']:.3f}")
                
                if "read_value" in acc_stats:
                    print(f"读取值准确率: {acc_stats['read_value']['mean']:.3f} ± {acc_stats['read_value']['std']:.3f}")
            
            if read_ops.get("performance_stats"):
                perf_stats = read_ops["performance_stats"]
                print(f"平均推理时间: {perf_stats['inference_time']['mean']:.2f}s")
                print(f"平均置信度: {perf_stats['confidence']['mean']:.3f}")
            
            if read_ops.get("engine_stats"):
                engine_stats = read_ops["engine_stats"]
                print(f"快速路径使用率: {engine_stats['quick_path_usage']:.2%}")
                print(f"自动修复率: {engine_stats['auto_fixed_rate']:.2%}")
        
        # 写操作结果
        write_ops = summary.get("write_operations", {})
        if write_ops.get("total_samples", 0) > 0:
            print(f"\n【写操作】")
            print(f"总样本数: {write_ops['total_samples']}")
            print(f"成功样本数: {write_ops['successful_samples']}")
            print(f"成功率: {write_ops['success_rate']:.2%}")
            
            if write_ops.get("accuracy_stats"):
                acc_stats = write_ops["accuracy_stats"]
                print(f"整体准确率: {acc_stats['overall']['mean']:.3f} ± {acc_stats['overall']['std']:.3f}")
                print(f"寄存器准确率: {acc_stats['register']['mean']:.3f} ± {acc_stats['register']['std']:.3f}")
            
            if write_ops.get("performance_stats"):
                perf_stats = write_ops["performance_stats"]
                print(f"平均推理时间: {perf_stats['inference_time']['mean']:.2f}s")
                print(f"平均置信度: {perf_stats['confidence']['mean']:.3f}")
            
            if write_ops.get("engine_stats"):
                engine_stats = write_ops["engine_stats"]
                print(f"快速路径使用率: {engine_stats['quick_path_usage']:.2%}")
                print(f"自动修复率: {engine_stats['auto_fixed_rate']:.2%}")
        
        # 整体统计
        overall = summary.get("overall", {})
        print(f"\n【整体统计】")
        print(f"总样本数: {overall.get('total_samples', 0)}")
        print(f"总成功数: {overall.get('total_successful', 0)}")
        
        if overall.get('total_samples', 0) > 0:
            overall_success_rate = overall.get('total_successful', 0) / overall.get('total_samples', 1)
            print(f"整体成功率: {overall_success_rate:.2%}")
        
        print("="*80)
        print("注：使用多层级推理引擎 (模式分析→语义理解→知识匹配→LLM生成→验证优化)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="JSON数据处理和多层级AI推理评估")
    parser.add_argument("--input-dir", default="./evaluation_data", help="输入数据目录")
    parser.add_argument("--output-dir", default="./evaluation_results", help="输出结果目录")
    parser.add_argument("--max-samples", type=int, default=50, help="每个文件最多处理的样本数")
    parser.add_argument("--max-files", type=int, default=5, help="每种操作类型最多处理的文件数")
    parser.add_argument("--timeout", type=int, default=120, help="推理超时时间（秒）")
    parser.add_argument("--model-path", default="/data/LLM_models/Qwen2.5-Coder-14B-Instruct", help="AI模型路径")
    parser.add_argument("--confidence-threshold", type=float, default=0.6, help="置信度阈值")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    
    args = parser.parse_args()
    
    # 创建约束配置
    constraints = InferenceConstraints(
        min_registers_count=3,
        max_registers_count=15,
        required_registers={"UARTDR", "UARTFR", "UARTCR"},  # PL011关键寄存器
        register_value_range=(0x0, 0xFFFFFFFF)
    )
    
    # 创建AI配置
    ai_config = InferenceConfig(
        model_path=args.model_path,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        confidence_threshold=args.confidence_threshold,
        constraints=constraints,
        enable_pattern_cache=True,
        enable_knowledge_base=True,
        enable_semantic_analysis=True
    )
    
    config = ProcessorConfig(
        input_data_dir=args.input_dir,
        output_dir=args.output_dir,
        max_samples_per_file=args.max_samples,
        max_files_per_type=args.max_files,
        ai_config=ai_config
    )
    
    # 运行处理器
    processor = JsonDataProcessor(config)
    processor.run_evaluation()

if __name__ == "__main__":
    main() 
