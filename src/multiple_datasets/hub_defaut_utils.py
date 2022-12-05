
def push_to_hub_using_whisper_template(trainer, finetuned_from, model_name, language):
    trainer.push_to_hub(**{
        "finetuned_from": finetuned_from,
        "tasks": "automatic-speech-recognition",
        "tags": ["whisper-event", "hf-asr-leaderboard"],
        "dataset": ["Common Voice 11.0"],
        "dataset_tags": ["mozilla-foundation/common_voice_11_0"],
        # "dataset_metadata": [{"config": language, "split": "test"}],
        "language": language,
        "model_name": model_name,
    })
