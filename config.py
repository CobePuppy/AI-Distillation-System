CONFIG = {
    # 'model_a_path': 'C:/Users/20267/Desktop/workflow/sample/model', # 备用路径：如果你有自己微调好的模型，可以把路径填在这
    'model_a_path': 'models_cache/qwen/Qwen2___5-0___5B-Instruct', # Model A 的路径：这里用的是 ModelScope 下下来的 0.5B 小模型，跑得快，测试逻辑够用了
    'model_b_path': 'models_cache/qwen/Qwen2___5-0___5B-Instruct', # Model B 的路径：同上，负责修正 Model A 的结果
    'model_c_api_key': 'sk-914903f674204aadad93fc372c289f4a',      # DeepSeek 的 API Key，Model C 是在线的大模型，用来当“老师”
    'model_c_api_url': 'https://api.deepseek.com',                 # API 接口地址，一般不用改
    'output_dir': 'results',                                       # 跑完的结果都扔这个文件夹里
    'input_data_dir': 'data'                                       # 输入数据放这
}
