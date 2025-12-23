import os
# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 关闭SSL验证
os.environ['CURL_CA_BUNDLE'] = ''
# 关闭代理
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

from config import CONFIG
from manager import WorkflowManager
from loader import load_data

def run_test():
    # 创建输出目录
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        
    # 初始化管理器
    manager = WorkflowManager(CONFIG)
    
    # 加载数据
    num_samples = 1 
    samples = load_data(split='validation', num_samples=num_samples)
    
    print(f"Running workflow on {len(samples)} samples...")
    
    for sample in samples:
        # 处理样本
        manager.process_sample(sample, max_loops=3)
        
    print("Test completed. Check the 'results' directory.")

if __name__ == "__main__":
    run_test()
