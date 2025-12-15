from .base import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelA(BaseModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"Loading Local Model A from {model_path}...")
        
        # 把本地模型和分词器加载进来，显存不够的话记得去 config.py 换个小点的模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, input_data: str, **kwargs) -> str:
        print(f"Model A processing input: {input_data[:50]}...")
        
        messages = [
            {"role": "user", "content": input_data}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Qwen 系列模型不需要 enable_thinking 参数，如果是 DeepSeek R1 这类带思考过程的模型才需要开
            # enable_thinking=True 
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=8192  # 生成长度限制，别太短也别太长
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 解析输出内容。这段逻辑主要是为了兼容带思考过程的模型（比如 DeepSeek R1），
        # 普通模型跑这段也没事，找不到标签就直接返回全部内容。
        try:
            # 找一下 </think> 标签在哪里，把思考过程和正文分开
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        # 返回主要内容，也可以选择返回包含思考过程的完整内容
        return content
