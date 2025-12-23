from datasets import load_dataset

# 加载数据
def load_data(split='validation', num_samples=5):
    """加载测试数据"""
    print(f"Loading dataset ({split} split)...")
    try:
        # 加载HF数据
        dataset = load_dataset("conll2003", split=split, trust_remote_code=True)
        
        samples = []
        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            tokens = item['tokens']
            text = " ".join(tokens)
            samples.append({
                'id': item['id'],
                'tokens': tokens,
                'ner_tags': item['ner_tags'],
                'text': text
            })
        return samples
    except Exception as e:
        print(f"Warning: Failed to load dataset ({e}). Using dummy data.")
        return [
            {
                'id': '0',
                'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
                'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0],
                'text': "EU rejects German call to boycott British lamb ."
            },
            {
                'id': '1',
                'tokens': ['Peter', 'Blackburn'],
                'ner_tags': [1, 1],
                'text': "Peter Blackburn"
            },
            {
                'id': '2',
                'tokens': ['BRUSSELS', '1996-08-22'],
                'ner_tags': [5, 0],
                'text': "BRUSSELS 1996-08-22"
            }
        ]

if __name__ == "__main__":
    # Test loading
    data = load_conll_samples(num_samples=2)
    for d in data:
        print(f"ID: {d['id']}")
        print(f"Text: {d['text']}")
        print("-" * 20)
