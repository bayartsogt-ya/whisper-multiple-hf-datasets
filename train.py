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
from multiple_datasets.dataset_utils import (
    merge_datasets,
    KEEP_CHARS)
from multiple_datasets.evaluate_utils import evaluate_and_save, get_compute_metrics_func
from multiple_datasets.data_collators import DataCollatorSpeechSeq2SeqWithPadding
from multiple_datasets.hub_default_utils import push_to_hub_using_whisper_template


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_datasets', default=None, help='dataset|config|splits,dataset|config|splits')
    parser.add_argument('--eval_datasets', default=None, help='dataset|config|splits,dataset|config|splits')
    parser.add_argument('--interleave', action='store_true', help='')
    parser.add_argument('--whisper-size', default='small')
    parser.add_argument('--language', default='mn,Mongolian', help='acronym,Full Language Name')
    parser.add_argument('--keep-chars', default=KEEP_CHARS, help='characters that would stay during preprocessing')
    parser.add_argument('--train-batch-size', default=32, type=int)
    parser.add_argument('--eval-batch-size', default=16, type=int)
    parser.add_argument('--max-steps', default=1000, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--version', default=1, type=int)

    # for reading and writing preprocessed dataset
    parser.add_argument('--read-from-preprocessed', action='store_true', help='if True, it will try to read from preprocessed dataset handle')
    parser.add_argument('--hf-username', type=str, required=True)


    args = parser.parse_args()
    
    show_argparse(args)
    lan, language = args.language.split(',')

    model_name = f'openai/whisper-{args.whisper_size}'
    output_dir = f"whisper-{args.whisper_size}-{lan}-{args.version}"
    
    print('model_name:', model_name)
    print('output_dir:', output_dir)

    ## Load
    config = WhisperConfig.from_pretrained(model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")
    
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False # not compatible with gradient checkpointing

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = get_compute_metrics_func(tokenizer)


    ## Preprocess
    train_ds = merge_datasets(
        args.train_datasets, args.interleave,
        args.keep_chars, feature_extractor, tokenizer,
        args.hf_username, args.read_from_preprocessed, args.num_workers)
    
    eval_ds = merge_datasets(
        args.eval_datasets, False,
        args.keep_chars, feature_extractor, tokenizer,
        args.hf_username, args.read_from_preprocessed, args.num_workers)


    # Train
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

    try:
        trainer.train()
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPTED! Starting evaluation with current state')
        trainer.is_in_train = False
        
    metrics = evaluate_and_save(trainer, tokenizer, feature_extractor)

    push_to_hub_using_whisper_template(
        args.train_datasets, args.hf_username, metrics, lan, output_dir
    )
