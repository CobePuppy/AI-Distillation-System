from modelscope import dataset_snapshot_download

try:
    dataset_dir = dataset_snapshot_download('conll2003', cache_dir='./data_cache')
    print(f"Dataset downloaded to: {dataset_dir}")
except Exception as e:
    print(f"Error: {e}")
