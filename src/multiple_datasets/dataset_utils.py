from datasets import load_dataset, interleave_datasets, concatenate_datasets

def merge_datasets(dataset_string: str, interleave: bool):
    ds_list = []
    for dataset_name in dataset_string.split(','):
        dataset_name, config, splits = dataset_name.split('|')
        for split in splits.split('+'):
            ds = load_dataset(dataset_name, config, split=split, use_auth_token=True)
            ds_list.append(ds)

    if interleave:
        ds = interleave_datasets(ds_list, seed=42)
    else: # just concat
        ds = concatenate_datasets(ds_list)
    
    return ds