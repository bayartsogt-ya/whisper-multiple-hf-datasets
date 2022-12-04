nvidia-smi

sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg

sudo apt-get install git-lfs
git-lfs -v
git lfs install

python3 -m venv venv
echo "source ~/$env_name/bin/activate" >> ~/.bashrc
bash

git clone https://github.com/bayartsogt-ya/whisper-multiple-hf-datasets
pip install -r whisper-multiple-hf-datasets/requirements.txt

git config --global credential.helper store
huggingface-cli login