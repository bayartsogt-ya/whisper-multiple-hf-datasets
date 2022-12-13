from copy import deepcopy
import torch
from transformers import WhisperForConditionalGeneration
from huggingface_hub import metadata_update


WHISPER_MAPPING = {
    "layers": "blocks",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "layers": "blocks",
    ".self_attn.q_proj": ".attn.query",
    ".self_attn.k_proj": ".attn.key",
    ".self_attn.v_proj": ".attn.value",
    ".self_attn_layer_norm": ".attn_ln",
    ".self_attn.out_proj": ".attn.out",
    ".encoder_attn.q_proj": ".cross_attn.query",
    ".encoder_attn.k_proj": ".cross_attn.key",
    ".encoder_attn.v_proj": ".cross_attn.value",
    ".encoder_attn_layer_norm": ".cross_attn_ln",
    ".encoder_attn.out_proj": ".cross_attn.out",
    "decoder.layer_norm.": "decoder.ln.",
    "encoder.layer_norm.": "encoder.ln_post.",
    "embed_tokens": "token_embedding",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "layer_norm": "ln_post",
}


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        print(f"{key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def convert_hf_whisper(hf_model_name_or_path: str, whisper_state_path: str):
    transformer_model = WhisperForConditionalGeneration.from_pretrained(hf_model_name_or_path)
    config = transformer_model.config

    # first build dims
    dims = {
        'n_mels': config.num_mel_bins,
        'n_vocab': config.vocab_size,
        'n_audio_ctx': config.max_source_positions,
        'n_audio_state': config.d_model,
        'n_audio_head': config.encoder_attention_heads,
        'n_audio_layer': config.encoder_layers,
        'n_text_ctx': config.max_target_positions,
        'n_text_state': config.d_model,
        'n_text_head': config.decoder_attention_heads,
        'n_text_layer': config.decoder_layers
    }

    state_dict = deepcopy(transformer_model.model.state_dict())
    state_dict = rename_keys(state_dict)

    torch.save({"dims": dims, "model_state_dict": state_dict}, whisper_state_path)


def push_to_hub_using_whisper_template(train_datasets, hf_username, metrics, language, output_dir):
    """Update the metadata of the README accordingly for `whisper-event`.

    Args:
        dataset_string (str): dataset_name|config|split
        hf_username (string): huggingface user handle
        metrics (dict): calculated metrics
        language (_type_): language acronym (e.g. `mn`)
        output_dir (_type_): huggingface model handle (e.g. `whisper-small-mn-1`)
    """
    
    metadata = {
        'license': 'apache-2.0',
        'tags': ['whisper-event', 'hf-asr-leaderboard', 'generated_from_multiple_datasets'],
        'language': language,
        'datasets': [dataset_handle.split('|')[0] for dataset_handle in train_datasets.split(',')],
        'metrics': ['wer', 'cer'],
        'model-index': [
            {
                'name': output_dir,
                'results': [
                    {
                        'task': {
                            'name': 'Automatic Speech Recognition',
                            'type': 'automatic-speech-recognition'
                        },
                        'dataset': {
                            'name': 'Common Voice 11.0',
                            'type': 'mozilla-foundation/common_voice_11_0',
                            'config': language,
                            'split': 'test',
                        },
                        'metrics': [
                            {
                                'name': 'Wer',
                                'type': 'wer',
                                'value': metrics['eval_wer']
                            },
                            {
                                'name': 'Cer',
                                'type': 'cer',
                                'value': metrics['eval_cer']
                            },
                        ],
                    }
                ]
            }
        ]
    }

    url = metadata_update(f"{hf_username}/{output_dir}", metadata, overwrite=True)
    print('URL to commit ->', url)
    return url
