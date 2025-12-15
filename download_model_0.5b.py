from modelscope import snapshot_download

model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./models_cache')
print(f"Model downloaded to: {model_dir}")
