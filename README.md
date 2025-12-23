# AI Distillation System

## 简介
CoNLL-2003 NER 任务工作流。
- **Model A**: 提取实体 (0.5B)
- **Model B**: 修正/评价 (0.5B)
- **Model C**: 专家评估 (DeepSeek)

## 文件说明
- `manager.py`: 核心流程控制 (A-B-C 循环)
- `run_dataset.py`: 跑 CoNLL 数据集测试
- `run_local.py`: 跑本地 txt 文件测试
- `download_resources.py`: 下载模型/数据
- `loader.py`: 加载数据工具
- `models/prompts.py`: 提示词集合

## 运行
1. 准备: `python download_resources.py`
2. 测数据: `python run_dataset.py`
3. 测本地: `python run_local.py`
