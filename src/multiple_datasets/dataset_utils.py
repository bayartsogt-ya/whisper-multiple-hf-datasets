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
    preprocess_func, prepare_dataset_func, # preprocessing functions
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
        ds = ds.map(merge_audio_mapper, batched=True, batch_size=10, remove_columns=list(ds.features), num_proc=num_workers)
        print('[IMPORTANT] dataset size AFTER merging:', ds.num_rows)

    ds = ds.map(preprocess_func, num_proc=num_workers)
    ds = ds.map(prepare_dataset_func)

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

    preprocess_func = get_preprocess_func(keep_chars)
    prepare_dataset = get_prepare_dataset_func(feature_extractor, tokenizer, merge_audio_to_max)

    ds_list = []
    for dataset_name in dataset_string.split(','):
        dataset_name, config, splits = dataset_name.split('|')
        config = config if config else None
        for split in splits.split('+'):
            ds = read_single_dataset(
                dataset_name, config, split,
                preprocess_func, prepare_dataset, 
                username, read_from_preprocessed, num_workers, merge_audio_to_max
            )
            ds_list.append(ds)

    if interleave:
        ds = interleave_datasets(ds_list, seed=42)
    else: # just concat
        ds = concatenate_datasets(ds_list)
    
    return ds


def get_preprocess_func(keep_chars):
    def preprocess_func(batch):
        batch[text_column] = re.sub(f"[^{keep_chars}]", "", batch[text_column].lower())
        return batch
    return preprocess_func


def get_prepare_dataset_func(feature_extractor: FeatureExtractionMixin, tokenizer: PreTrainedTokenizer, merge_audio_to_max: bool):
    """_summary_

    Args:
        feature_extractor (FeatureExtractionMixin): _description_
        tokenizer (PreTrainedTokenizer): _description_
        merge_audio_to_max (bool): if `True`, we can assume that `merge_audio_mapper` applied and `audio` column no longer exists.
    """
    def prepare_dataset(batch):
        audio = batch if merge_audio_to_max else batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch[text_column]).input_ids
        return batch

    return prepare_dataset


def merge_audio_mapper(batch):
    bs = len(batch[text_column])
    result = {'array': [], text_column: [], 'sampling_rate': []}
    list_arr, list_text, total = [], [], 0
    for i in range(bs + 1):
        if i == bs or total + batch[audio_column][i]['array'].shape[0] / 16_000 > 30.:
            result['array'].append(np.concatenate(list_arr))
            result[text_column].append(' '.join(list_text))
            result['sampling_rate'].append(16_000)
            list_arr, list_text, total = [], [], 0
        if i < bs:
            list_arr.append(batch[audio_column][i]['array'])
            list_text.append(batch[text_column][i])
            total += batch[audio_column][i]['array'].shape[0] / 16_000
    return result

