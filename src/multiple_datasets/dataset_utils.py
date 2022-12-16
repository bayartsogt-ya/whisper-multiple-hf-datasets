import re
import numpy as np
from datasets import load_dataset, interleave_datasets, concatenate_datasets, Audio, Dataset
from transformers import PreTrainedTokenizer
from transformers.feature_extraction_utils import FeatureExtractionMixin


text_column_names = set(['sentence', 'transcription', 'transciption']) # possible text column choices
audio_column = 'audio'
text_column = 'transcription'
MAX_AUDIO_DURATION = 30 # because of whisper model input is 30 second, TODO: refer to paper
KEEP_CHARS = " абвгдеёжзийклмноөпрстуүфхцчшъыьэюя"
DEFAULT_SAMPLING_RATE = 16_000


def get_preprocessed_dataset_name(dataset_name, config, split, prefix):
    preprocessed_dataset_name = [prefix]
    preprocessed_dataset_name.append(dataset_name.split('/')[-1])
    if config: preprocessed_dataset_name.append(config)
    preprocessed_dataset_name.append(split)
    preprocessed_dataset_name = '-'.join(preprocessed_dataset_name)
    return preprocessed_dataset_name


def read_single_dataset(
    dataset_name: str, config: str, split: str, # dataset info
    keep_chars: str, feature_extractor: FeatureExtractionMixin, tokenizer: PreTrainedTokenizer,
    username: str, read_from_preprocessed: bool, # read from preprocessed dataset
    num_workers: int, merge_audio_to_max: bool
):
    # pp -> Pre-Processed
    # mpp -> Merge audios to 30 sec and Pre-Processed
    prefix = ('m' if merge_audio_to_max else '') + 'pp'
    preprocessed_dataset_name = get_preprocessed_dataset_name(dataset_name, config, split, prefix)

    if read_from_preprocessed:
        # read from preprocessed dataset repo
        # TODO: check not just name but if all the metadata args matches our expectation
        # cached dataset will always read with `train` split
        print(f'--------> Reading from HF Hub {username + "/" + preprocessed_dataset_name}')
        try:
            return load_dataset(username + '/' + preprocessed_dataset_name, split='train', use_auth_token=True)
        except FileNotFoundError as e:
            print(f'Could not find the {username + "/" + preprocessed_dataset_name}. So creating it from the scratch.')

    ds = load_dataset(dataset_name, config, split=split, use_auth_token=True)

    # use same name for all datasets
    if text_column not in ds.column_names:
        prev_text_col = set(ds.column_names).intersection(text_column_names).pop()
        ds = ds.rename_column(prev_text_col, text_column)

    # remove unnecessary columns
    remove_cols = list(set(ds.column_names) - set(['audio', text_column]))
    ds = ds.remove_columns(remove_cols)

    # preprocess
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # make preprocessing here
    assert type(ds) == Dataset
    if merge_audio_to_max:
        print('[IMPORTANT] dataset size BEFORE merging:', ds.num_rows)
        mapper = get_batch_mapper_merging_max_duration(keep_chars, feature_extractor, tokenizer)
        ds = ds.map(mapper, batched=True, batch_size=30, remove_columns=list(ds.features), num_proc=2)
        print('[IMPORTANT] dataset size AFTER merging:', ds.num_rows)
    else:
        mapper = get_mapper(keep_chars, feature_extractor, tokenizer)
        ds = ds.map(get_mapper, num_proc=num_workers)

    # write a preprocessed dataset to huggingface hub
    print(f'--------> Writing to HF Hub {preprocessed_dataset_name}')
    ds.push_to_hub(preprocessed_dataset_name, private=True)
    
    return ds


def merge_datasets(
    dataset_string: str, interleave: bool,
    keep_chars: str, feature_extractor: FeatureExtractionMixin, tokenizer: PreTrainedTokenizer,
    username: str, read_from_preprocessed: bool, num_workers: int, merge_audio_to_max: bool) -> Dataset:
    """Read multiple datasets and upload preprocessed dataset for reading later on.

    Args:
        dataset_string (str): dataset_name|config|split
        interleave (bool): whether to interleave datasets, if false concatnate
        keep_chars (str): characters will be kept during preprocessing (it always apply lower_case)
        feature_extractor (FeatureExtractionMixin): Feature extractor (e.g. WhisperFeatureExtractor)
        tokenizer (PreTrainedTokenizer): Tokenizer (e.g. WhisperTokenizer)
        username (str): huggingface handle for reading preprocessed dataset
        read_from_preprocessed (bool): whether to lead from preprocessed or not
        num_workers (int): number of workers to be used during `map` processing
        merge_audio_to_max (bool): if True, then it will merge audios to `MAX_AUDIO_DURATION`

    Returns:
        dataset (Dataset): preprocesed and merged dataset
    """
    ds_list = []
    for dataset_name in dataset_string.split(','):
        dataset_name, config, splits = dataset_name.split('|')
        config = config if config else None
        for split in splits.split('+'):
            ds = read_single_dataset(
                dataset_name, config, split,
                keep_chars, feature_extractor, tokenizer,
                username, read_from_preprocessed, num_workers, merge_audio_to_max
            )
            ds_list.append(ds)

    if interleave:
        ds = interleave_datasets(ds_list, seed=42)
    else: # just concat
        ds = concatenate_datasets(ds_list)
    
    return ds


def get_batch_mapper_merging_max_duration(
    keep_chars: str,
    feature_extractor: FeatureExtractionMixin,
    tokenizer: PreTrainedTokenizer):

    def text_column_normalize(text):
        return re.sub(f"[^{keep_chars}]", "", text.lower())

    def mapper(batch):
        bs = len(batch[text_column])
        result = {'input_features': [], 'labels': []}
        list_arr, list_text, total = [], [], 0
        for i in range(bs + 1):
            if i == bs or total + batch[audio_column][i]['array'].shape[0] / DEFAULT_SAMPLING_RATE > MAX_AUDIO_DURATION:
                if total == 0: continue # because it could be evenly distributed when i == bs
                result['input_features'].append(
                    feature_extractor(np.concatenate(list_arr), sampling_rate=DEFAULT_SAMPLING_RATE).input_features[0]
                )
                result['labels'].append(
                    tokenizer(text_column_normalize(' '.join(list_text))).input_ids
                )
                list_arr, list_text, total = [], [], 0
            if i < bs:
                duration = batch[audio_column][i]['array'].shape[0] / DEFAULT_SAMPLING_RATE
                if duration > MAX_AUDIO_DURATION: continue
                total += duration
                list_arr.append(batch[audio_column][i]['array'])
                list_text.append(batch[text_column][i])
        return result
    return mapper


def get_mapper(
    keep_chars: str,
    feature_extractor: FeatureExtractionMixin,
    tokenizer: PreTrainedTokenizer):

    def text_column_normalize(text):
        return re.sub(f"[^{keep_chars}]", "", text.lower())

    def mapper(example):
        return {
            'input_features': feature_extractor(example[audio_column]['array'], sampling_rate=DEFAULT_SAMPLING_RATE).input_features[0],
            'labels': tokenizer(text_column_normalize(example[text_column])).input_ids
        }
    
    return mapper

