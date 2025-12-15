import os
import glob
from config import CONFIG
from workflow_manager import WorkflowManager

def main():
    # 先把管家（WorkflowManager）请出来，配置都给它
    manager = WorkflowManager(CONFIG)
    
    # 去 data 目录翻翻看有没有 .txt 文件要处理
    input_dir = CONFIG['input_data_dir']
    input_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not input_files:
        print(f"No input files found in {input_dir}. Please add .txt files.")
        return

    # 挨个处理找到的文件
    for file_path in input_files:
        manager.process_file(file_path)

if __name__ == "__main__":
    main()
