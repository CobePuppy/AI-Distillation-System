from .base import BaseModel
from .prompts import get_model_b_critique_prompt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelB(BaseModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"[调试信息] 正在加载本地模型 B，路径：{model_path}...")
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        # 生成修正文本
        print(f"[调试信息] 模型 B 正在生成...")
        
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
            max_new_tokens=8192
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        return content

    def critique(self, sent: str, prediction: str, previous_feedback: str = None) -> str:
        print(f"[调试信息] 模型 B 正在进行批评...")
        prompt = get_model_b_critique_prompt(sent, prediction, previous_feedback)
        
        return self.generate(prompt)

    def construct_correction_prompt(self, input_params: str, result_a: str, result_c: str) -> str:
        """
        拼装 Prompt：把 原始输入 + A的输出 + C的参考答案 组合起来，让 B 照着 C 的样子去改 A
        """
        return f"""
        Original Input: {input_params}
        
        Model A Output: {result_a}
        
        Model C Output (Reference): {result_c}
        
        Task: Correct Model A's output to match the quality/style of Model C's output.
        """
