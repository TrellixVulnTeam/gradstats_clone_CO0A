 conda activate /home/ubuntu/anaconda3/envs/pytorch_latest_p37
 cd /fsx/code/gradstats/BERT
  pip install -r requirements.txt
 WORK_DIR=`mktemp -d`
 cd $WORK_DIR
 git clone https://github.com/NVIDIA/apex
 cd apex
 pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ &> /fsx/logs/log_pytorch_latest_p37
 pip install tqdm tensorboard yacs

# Copy conda env to fsx
 #  wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
# chmod +x Miniconda3-py37_4.10.3-Linux-x86_64.sh
#  ./Miniconda3-py37_4.10.3-Linux-x86_64.sh
#  cp -r /home/ubuntu/anaconda3/envs/pytorch_latest_p37 /fsx/conda/envs/