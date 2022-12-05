import argparse
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer)

# # local
from multiple_datasets.utils import show_argparse
from multiple_datasets.dataset_utils import get_prepare_dataset_func, merge_datasets, preprocess_func
from multiple_datasets.evaluate_utils import get_compute_metrics_func
from multiple_datasets.data_collators import DataCollatorSpeechSeq2SeqWithPadding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_datasets', default=None, help='dataset|config|splits,dataset|config|splits')
    parser.add_argument('--eval_datasets', default=None, help='dataset|config|splits,dataset|config|splits')
    parser.add_argument('--interleave', action='store_true', help='')
    parser.add_argument('--whisper-size', default='small')
    parser.add_argument('--language', default='mn,Mongolian', help='acronym,Full Language Name')
    parser.add_argument('--train-batch-size', default=32, type=int)
    parser.add_argument('--eval-batch-size', default=16, type=int)
    parser.add_argument('--max-steps', default=1000, type=int)
    parser.add_argument('--version', default=1, type=int)
    parser.add_argument('--num-workers', default=8, type=int)

    args = parser.parse_args()
    
    show_argparse(args)

    model_name = f'openai/whisper-{args.whisper_size}'
    output_dir = f"whisper-{args.whisper_size}-{args.language.split(',')[0]}-{args.version}"

    print('model_name:', model_name)
    print('output_dir:', output_dir)


    train_ds = merge_datasets(args.train_datasets, args.interleave)
    print(train_ds)
    
    eval_ds = merge_datasets(args.eval_datasets, False)
    print(eval_ds)

    
    config = WhisperConfig.from_pretrained(model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=args.language.split(',')[1], task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language=args.language.split(',')[1], task="transcribe")
    
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False # not compatible with gradient checkpointing

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = get_compute_metrics_func(tokenizer)

    # data preprocessing
    prepare_dataset_func = get_prepare_dataset_func(feature_extractor, tokenizer)
    train_ds = train_ds.map(preprocess_func, num_proc=args.num_workers)
    eval_ds = eval_ds.map(preprocess_func, num_proc=args.num_workers)
    train_ds = train_ds.map(prepare_dataset_func, num_proc=args.num_workers)
    eval_ds = eval_ds.map(prepare_dataset_func, num_proc=args.num_workers)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        remove_unused_columns=False, # important when we use set_transform

        #
        dataloader_num_workers=args.num_workers
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    
    metrics = trainer.evaluate(
        metric_key_prefix="eval",
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model()
    trainer.save_state()

    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    
    trainer.push_to_hub(**{
        "finetuned_from": model_name,
        "tasks": "automatic-speech-recognition",
        "tags": "whisper-event",
        "dataset": "mozilla-foundation/common_voice_11_0 mn",
        "language": "mn",
        "model_name": output_dir,
    })

