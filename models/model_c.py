from .base import BaseModel
from .prompts import get_model_c_evaluation_prompt
from openai import OpenAI

class ModelC(BaseModel):
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.api_url = api_url
        print("[调试信息] 已初始化在线模型 C (DeepSeek)")
        
        # 初始化客户端
        self.client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )

    def generate(self, input_data: str, **kwargs) -> str:
        print(f"[调试信息] 模型 C 正在处理输入：{input_data[:50]}...")
        
        messages = [
            {"role": "user", "content": input_data}
        ]

        # 调用API
        completion_generator = self.client.chat.completions.create(
            model="deepseek-reasoner", 
            messages=messages,
            stream=True
        )
        
        content, thinking_content = '', ''
        for chunk in completion_generator:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                thinking_content += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                content += delta.content

        return content

    def evaluate(self, sent: str, label: str, prediction: str) -> str:
        print(f"[调试信息] 模型 C 正在进行评估...")
        
        prompt = get_model_c_evaluation_prompt(sent, label, prediction)
        
        return self.generate(prompt)
