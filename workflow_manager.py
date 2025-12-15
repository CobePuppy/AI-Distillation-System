import os
from models.model_a import ModelA
from models.model_b import ModelB
from models.model_c import ModelC

class WorkflowManager:
    def __init__(self, config):
        self.model_a = ModelA(config['model_a_path'])
        self.model_b = ModelB(config['model_b_path'])
        self.model_c = ModelC(config['model_c_api_key'])
        self.output_dir = config['output_dir']
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_file(self, input_file_path: str):
        print(f"--- Starting Workflow for {input_file_path} ---")
        
        # 1. 先把文件内容读出来，这就是咱们的原始输入
        with open(input_file_path, 'r', encoding='utf-8') as f:
            input_params = f.read()
            
        self.process_text(input_params, os.path.basename(input_file_path))

    def process_text(self, input_text: str, source_name: str = "unknown"):
        print(f"--- Processing text from {source_name} ---")
        
        # 2. 让 Model A 先试着做一下（初稿）
        result_a = self.model_a.generate(input_text)
        print(f"Model A Result: {result_a}")
        
        # 3. 让 Model C（老师傅）给个标准答案
        result_c = self.model_c.generate(input_text)
        print(f"Model C Result: {result_c}")
        
        # 4. 让 Model B 看着老师傅的答案，帮 Model A 改改作业
        prompt_b = self.model_b.construct_correction_prompt(input_text, result_a, result_c)
        result_b = self.model_b.generate(prompt_b)
        print(f"Model B Result: {result_b}")
        
        # 5. 把大家的工作成果都存起来，留个底
        self.save_results(source_name, input_text, result_a, result_c, result_b)
        
    def save_results(self, base_name, input_params, res_a, res_c, res_b):
        output_path = os.path.join(self.output_dir, f"result_{base_name}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Input:\n{input_params}\n\n")
            f.write(f"Model A Output:\n{res_a}\n\n")
            f.write(f"Model C Output (Reference):\n{res_c}\n\n")
            f.write(f"Model B Output (Correction):\n{res_b}\n\n")
            
        print(f"Results saved to {output_path}")
