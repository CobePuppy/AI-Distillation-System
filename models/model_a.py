from .base import BaseModel
from .prompts import get_model_a_prompt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelA(BaseModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"[调试信息] 正在加载本地模型 A，路径：{model_path}...")
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, input_data: str, **kwargs) -> str:
        print(f"[调试信息] 模型 A 正在处理输入：{input_data[:50]}...")
        
        # 构造JSON提示词
        prompt = get_model_a_prompt(input_data)

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        return content
