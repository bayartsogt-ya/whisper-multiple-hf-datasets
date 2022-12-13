import evaluate

def get_compute_metrics_func(tokenizer):
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
        cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

        return {
            "wer": wer,
            "cer": cer
        }
    
    return compute_metrics


def evaluate_and_save(trainer, tokenizer, feature_extractor):
    training_args = trainer.args
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
    trainer.model.config.save_pretrained(training_args.output_dir)
    trainer.push_to_hub()

    return metrics
