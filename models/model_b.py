from .base import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelB(BaseModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"Loading Local Model B from {model_path}...")
        
        # 初始化本地模型和分词器，注意路径要对，显存不够的话去 config.py 换个小点的模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        # Model B 的主要任务：基于提示词生成修正后的文本
        print(f"Model B correcting...")
        
        messages = [
            {"role": "user", "content": prompt}
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
            max_new_tokens=8192
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # 解析输出内容。这段逻辑主要是为了兼容带思考过程的模型（比如 DeepSeek R1），
        # 普通模型跑这段也没事，找不到标签就直接返回全部内容。
        # 151668 是 </think> 的 token id，不同模型可能不一样，注意检查。
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return content

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
