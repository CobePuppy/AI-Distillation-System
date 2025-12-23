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
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            input_params = f.read()
            
        # 处理文件输入
        self.process_text(input_params, os.path.basename(input_file_path))

    def process_sample(self, sample, max_loops=3):
        sent = sample['text']
        label = str(sample['ner_tags']) # 转换标签格式
        source_name = sample.get('id', 'unknown')
        
        print(f"--- Processing sample {source_name} ---")
        
        # 1. A Extractor
        pred_a = self.model_a.generate(sent)
        print(f"Model A Prediction: {pred_a}")
        
        # 2. B Initial Judgment (on sent only)
        initial_b = self.model_b.generate(f"Analyze the named entities in this sentence: {sent}")
        print(f"Model B Initial Judgment: {initial_b}")
        
        # 3. Loop
        feedback_c = None
        final_feedback = ""
        
        for i in range(max_loops):
            print(f"--- Loop {i+1}/{max_loops} ---")
            
            # B生成评价
            critique_b = self.model_b.critique(sent, pred_a, feedback_c)
            print(f"Model B Critique (Round {i+1}): {critique_b}")
            
            # C评估预测
            critique_c = self.model_c.evaluate(sent, label, pred_a)
            print(f"Model C Critique (Round {i+1}): {critique_c}")
            
            # 更新反馈
            feedback_c = critique_c
            final_feedback = critique_b 
            
        # 4. Save Results
        self.save_results(source_name, sent, pred_a, label, final_feedback, feedback_c)

    def process_text(self, input_text: str, source_name: str = "unknown"):
        # 无标签处理
        print("Processing text without label (Simulated run)")
        sample = {'text': input_text, 'ner_tags': "Unknown (No Label Provided)", 'id': source_name}
        self.process_sample(sample, max_loops=1)
        
    def save_results(self, base_name, input_params, res_a, label, res_b, res_c):
        output_path = os.path.join(self.output_dir, f"result_{base_name}.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Input:\n{input_params}\n\n")
            f.write(f"Label:\n{label}\n\n")
            f.write(f"Model A Output:\n{res_a}\n\n")
            f.write(f"Model B Final Feedback:\n{res_b}\n\n")
            f.write(f"Model C Final Feedback:\n{res_c}\n\n")
            
        print(f"Results saved to {output_path}")
