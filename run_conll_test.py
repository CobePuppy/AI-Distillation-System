import os
# 换个国内的镜像源，不然 Hugging Face 连不上会急死人
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 测试环境先把 SSL 验证关了，免得报握手错误
os.environ['CURL_CA_BUNDLE'] = ''
# 把代理关了，防止干扰本地连接
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

from config import CONFIG
from workflow_manager import WorkflowManager
from conll_loader import load_conll_samples

def run_test():
    # 确保结果目录存在，没有就建一个
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        
    # 召唤工作流管理器
    manager = WorkflowManager(CONFIG)
    
    # 加载数据，先拿 1 条跑跑看，跑通了再加量
    num_samples = 1 
    samples = load_conll_samples(split='validation', num_samples=num_samples)
    
    print(f"Running workflow on {len(samples)} CoNLL-2003 samples...")
    
    for sample in samples:
        sample_id = sample['id']
        text = sample['text']
        
        # 构造提示词，告诉模型我们要干啥（NER任务）
        prompt = (
            "Task: Named Entity Recognition (NER)\n"
            "Please identify the named entities in the following text. "
            "Classify them into PER (Person), ORG (Organization), LOC (Location), MISC (Miscellaneous).\n\n"
            f"Text: {text}\n\n"
            "Output:"
        )
        
        # 起个文件名，别带怪字符
        source_name = f"conll_val_{sample_id}.txt"
        
        # 开工！
        manager.process_text(prompt, source_name)
        
    print("Test completed. Check the 'results' directory.")

if __name__ == "__main__":
    run_test()
