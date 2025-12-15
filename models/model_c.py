from .base import BaseModel
from openai import OpenAI

class ModelC(BaseModel):
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.api_url = api_url
        print("Initialized Online Model C (DeepSeek)")
        
        # 初始化 OpenAI 客户端，其实是连的 DeepSeek 的服务
        self.client = OpenAI(
            base_url=self.api_url,
            api_key=self.api_key,
        )

    def generate(self, input_data: str, **kwargs) -> str:
        print(f"Model C processing input: {input_data[:50]}...")
        
        messages = [
            {"role": "user", "content": input_data}
        ]

        # 调 API 拿结果，这里用的是 deepseek-reasoner，也就是 R1，比较聪明
        completion_generator = self.client.chat.completions.create(
            model="deepseek-reasoner", 
            messages=messages,
            stream=True
            # extra_body={
            #     "enable_thinking": True, # 如果 API 需要显式开启思考模式，就把这个注释解开
            # },
        )
        
        content, thinking_content = '', ''
        for chunk in completion_generator:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                thinking_content += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                content += delta.content

        # print("thinking content:", thinking_content)
        return content
