 conda activate /home/ubuntu/anaconda3/envs/pytorch_latest_p37
 cd /fsx/code/gradstats/BERT
  pip install -r requirements.txt
 WORK_DIR=`mktemp -d`
 cd $WORK_DIR
 git clone https://github.com/NVIDIA/apex
 cd apex
 pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ &> log_pytorch_latest_p37
 pip install tqdm tensorboard yacs
 