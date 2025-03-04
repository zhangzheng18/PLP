import torch
import json
import threading
import time
import uuid
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

class InferenceSystem:
    """
    高效的模型推理系统，支持单次加载多次推理
    通过共享内存字典实现结果存储（线程安全版本）
    """
    def __init__(self, model_path="/home/zhangzheng/work1/llama"):
        # 硬件配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing model on {self.device}...")
        
        # 模型加载
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 使用内存映射加快大模型加载
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        # 创建线程安全的共享内存空间
        self.shared_memory = {
            'input_queue': [],
            'results': {},
            'lock': threading.Lock()
        }
        
        # 启动后台推理线程
        self.inference_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.inference_thread.start()

    def _create_prompt(self, input_data):
        """构造标准化提示模板"""
        return (
            f"### Task: Predict MAC and PHY registers\n"
            f"### NIC State and Configuration:\n{input_data}\n\n"
            f"### Predict the following:\n"
        )

    def _processing_loop(self):
        """后台处理循环"""
        while True:
            with self.shared_memory['lock']:
                if self.shared_memory['input_queue']:
                    request_id, input_data = self.shared_memory['input_queue'].pop(0)
                else:
                    request_id, input_data = None, None
            
            if request_id:
                try:
                    # 构造提示词
                    prompt = self._create_prompt(json.dumps(input_data, indent=2))
                    
                    # 编码输入
                    inputs = self.tokenizer(
                        prompt,
                        truncation=True,
                        max_length=512,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad(), torch.amp.autocast("cuda"):
                        outputs = self.model.generate(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_length=512,
                            temperature=0.3,
                            top_p=0.9,
                            do_sample=True,
                        )

                    
                    # 解码结果
                    result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    with self.shared_memory['lock']:
                        self.shared_memory['results'][request_id] = {
                            'status': 'completed',
                            'prediction': result
                        }
                except Exception as e:
                    with self.shared_memory['lock']:
                        self.shared_memory['results'][request_id] = {
                            'status': 'error',
                            'message': str(e)
                        }
            else:
                time.sleep(0.1)  # 减少CPU占用

    def submit_request(self, context):
        """
        提交推理请求到共享内存队列
        返回唯一请求ID用于结果查询
        """
        request_id = str(uuid.uuid4())
        
        with self.shared_memory['lock']:
            self.shared_memory['input_queue'].append((request_id, context))
            self.shared_memory['results'][request_id] = {'status': 'pending'}
        
        return request_id

    def get_result(self, request_id, timeout=30):
        """
        从共享内存获取推理结果
        支持超时机制
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.shared_memory['lock']:
                result = self.shared_memory['results'].get(request_id)
                
            if result['status'] in ('completed', 'error'):
                return result
            time.sleep(0.1)
        
        raise TimeoutError("Request timed out after 30 seconds")
    
def infer_response(request_id):
    """
    获取推理请求的结果
    使用请求ID从推理系统中提取结果
    """
    try:
        # 获取结果（阻塞式）
        result = inference_system.get_result(request_id)
        if result['status'] == 'completed':
            return {
                'status': 'completed',
                'prediction': result['prediction']
            }
        else:
            return {
                'status': 'error',
                'message': result.get('message', 'Unknown error')
            }
    except TimeoutError as e:
        return {
            'status': 'error',
            'message': str(e)
        }

# 初始化推理系统（在应用启动时执行）
inference_system = InferenceSystem()

# 使用示例
if __name__ == "__main__":
    # 模拟输入数据
    test_context = {
        "register_map": {
            "CR0": 0xA001,
            "PHY_CTRL": 0x1F
        },
        "link_status": "up",
        "speed": "1Gbps"
    }
    
    # 提交请求
    request_id = inference_system.submit_request(test_context)
    
    try:
        # 获取结果（阻塞式）
        result = inference_system.get_result(request_id)
        if result['status'] == 'completed':
            print("推理结果：")
            print(result['prediction'])
        else:
            print("推理错误：", result['message'])
    except TimeoutError as e:
        print(e)