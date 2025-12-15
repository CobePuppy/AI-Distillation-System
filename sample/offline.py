from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./model" # 使用本地路径或者huggingface模型名称

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 准备模型输入
prompt = '''Extract entities and their types from the document. Output a JSON object whose keys are entity names and whose values are the corresponding entity types.
document: SpaceX was founded by Elon Musk in 2002.'''
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,   # 对话历史
    tokenize=False, # 是否分词并转换为相应id
    add_generation_prompt=True, # 是否在末尾添加“助手”标记提示角色
    enable_thinking=True    # 是否允许思考：设为False将添加思考的开始与结束标记来跳过思考过程
)
# print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)  # 将若干输入文本分词并转化为pytorch张量
# {"input_ids": <二维张量>, "attention_mask": <二维张量>}
# print(model_inputs)

# 执行文本补全
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
# generated_ids 是一个二维张量
# print(generated_ids)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() # 截取补全的词元id并转化为列表

# 解析思考内容
try:
    # 找到从右往左第一个151668（</think>）的正向索引（+1）
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)