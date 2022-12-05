import re
from datasets import load_dataset, interleave_datasets, concatenate_datasets, Audio

audio_column_names = set(['sentence', 'transcription'])
text_column = 'transcription'
keep_chars = " абвгдеёжзийклмноөпрстуүфхцчшъыьэюя"

def merge_datasets(dataset_string: str, interleave: bool):
    ds_list = []
    for dataset_name in dataset_string.split(','):
        dataset_name, config, splits = dataset_name.split('|')
        config = config if config else None
        for split in splits.split('+'):
            ds = load_dataset(dataset_name, config, split=split, use_auth_token=True)

            # use same name for all datasets
            if text_column not in ds.column_names:
                prev_text_col = set(ds.column_names).intersection(audio_column_names).pop()
                ds = ds.rename_column(prev_text_col, text_column)

            # remove unnecessary columns
            remove_cols = list(set(ds.column_names) - set(['audio', text_column]))
            ds = ds.remove_columns(remove_cols)

            # preprocess
            ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
            ds_list.append(ds)

    if interleave:
        ds = interleave_datasets(ds_list, seed=42)
    else: # just concat
        ds = concatenate_datasets(ds_list)
    
    return ds


def preprocess_func(batch):
    batch[text_column] = re.sub(f"[^{keep_chars}]", "", batch[text_column].lower())
    return batch

def get_prepare_dataset_func(feature_extractor, tokenizer):
    # # no transform everything is in 
    def prepare_dataset(batch):
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch[text_column]).input_ids
        return batch

    return prepare_dataset
