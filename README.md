# whisper-multiple-hf-datasets
Whisper fine-tuning event script to use multiple hf datasets


### Train using multiple huggingface models

This works for only **Mongolian** for now but feel free to change
`multiple_datasets.dataset_utils.keep_chars` variable for preprocessing.

```bash
python train.py \
    --train_datasets "mozilla-foundation/common_voice_11_0|mn|train+validation,google/fleurs|mn_mn|train+validation" \
    --eval_datasets "mozilla-foundation/common_voice_11_0|mn|test" \
    --whisper-size "small" \
    --language "mn,Mongolian" \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --max-steps 15000 \
    --num-workers 8 \
    --read-from-preprocessed \
    --hf-username 'your-huggingface-name' \
    --version 1
```

### Convert Huggingface model to `whisper` model

```python
# install multiple_datasets
!pip install git+https://github.com/bayartsogt-ya/whisper-multiple-hf-datasets.git

from multiple_datasets.hub_default_utils import convert_hf_whisper
model_name_or_path = 'openai/whisper-tiny'
whisper_checkpoint_path = './whisper-tiny-checkpoint.pt'
convert_hf_whisper(model, whisper_checkpoint_path)

# now transcribe
import whisper
model = whisper.load_model(whisper_model_path)
result = model.transcribe('loooong_audio_path.wav') # probably longer than 10 min? hour?
print(result['text'])
```