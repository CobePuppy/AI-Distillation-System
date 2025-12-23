import os
import glob
from config import CONFIG
from manager import WorkflowManager

def main():
    # 初始化管理器
    manager = WorkflowManager(CONFIG)
    
    # 查找输入文件
    input_dir = CONFIG['input_data_dir']
    input_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not input_files:
        print(f"No input files found in {input_dir}. Please add .txt files.")
        return

    # 处理文件
    for file_path in input_files:
        manager.process_file(file_path)

if __name__ == "__main__":
    main()
