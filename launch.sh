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

pip install -r requirements.txt
pip install -e .

git config --global credential.helper store
huggingface-cli login
