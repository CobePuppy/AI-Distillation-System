from openai import OpenAI

# 建立连接
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your api_key",  # 在阿里千问官网获取API KEY
)

# 准备模型输入
prompt = '''Extract entities and their types from the document. Output a JSON object whose keys are entity names and whose values are the corresponding entity types.
document: SpaceX was founded by Elon Musk in 2002.'''
messages = [
    {"role": "user", "content": prompt}
]

# 调用API获取模型补全
completion_generator = client.chat.completions.create(
    model="qwen3-235b-a22b",
    temperature=0,
    max_tokens=16384,
    messages=messages,
    stream=True,
    extra_body={
        "enable_thinking": True,
        # "thinking_budge": 50,
    },
)
content, thinking_content = '', ''
for chunk in completion_generator:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        thinking_content += delta.reasoning_content
    if hasattr(delta, "content") and delta.content:
        content += delta.content

print("thinking content:", thinking_content)
print("content:", content)