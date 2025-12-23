import os
from modelscope import snapshot_download, dataset_snapshot_download

def download_all():
    print("--- 开始下载资源 ---")
    
    # 下载数据集
    print("\n正在下载 CoNLL-2003 数据集...")
    try:
        dataset_dir = dataset_snapshot_download('conll2003', cache_dir='./data_cache')
        print(f"数据集已就位: {dataset_dir}")
    except Exception as e:
        print(f"数据集下载出错了: {e}")

    # 下载模型
    print("\n正在下载 Qwen2.5-0.5B-Instruct 模型...")
    try:
        model_05b_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models_cache')
        print(f"0.5B 模型已就位: {model_05b_dir}")
    except Exception as e:
        print(f"0.5B 模型下载出错了: {e}")

    # 备用模型
    # print("\n正在下载 Qwen2.5-7B-Instruct 模型...")
    # try:
    #     model_7b_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./models_cache')
    #     print(f"7B 模型已就位: {model_7b_dir}")
    # except Exception as e:
    #     print(f"7B 模型下载出错了: {e}")
        
    print("\n--- 资源下载完成 ---")

if __name__ == "__main__":
    download_all()
