def get_model_a_prompt(input_data):
    return f"""你是一个信息抽取系统。
请从下面的句子中提取实体，并以JSON格式返回。
JSON应包含 "person"（人物）、"organization"（组织）、"location"（地点）、"misc"（其他）等键（如果存在）。
示例格式：{{"person": ["Elon Musk"], "organization": ["Tesla"], "location": [], "misc": []}}

句子：{input_data}

JSON输出："""

def get_model_b_critique_prompt(sent, prediction, previous_feedback=None):
    prompt = f"""你是一个评论家（Critic）。
句子：{sent}
预测结果：{prediction}

"""
    if previous_feedback:
        prompt += f"之前的专家反馈（请从中学习）：{previous_feedback}\n"
        
    prompt += "请提供你的批评意见，并给出如何改进预测的指令。"
    return prompt

def get_model_c_evaluation_prompt(sent, label, prediction):
    return f"""你是一位专家评估员。
原始句子：{sent}
真实标签（Ground Truth）：{label}
模型预测：{prediction}

请根据真实标签对预测结果进行评估。
请提供思维链（Chain of Thought）推理过程以及最终的评估结果。
指出任何缺失的实体、错误的类型或多余的实体。
为模型提供建设性的反馈意见。
"""
