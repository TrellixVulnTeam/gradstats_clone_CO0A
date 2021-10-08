
nodes=$(cat $1 | wc -l)
rank=0
for host in $hosts; do

	CMD="/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=${rank} --master_addr=compute-st-p4d24xlarge-1 --master_port=12244 /fsx/code/gradstats/BERT//run_pretraining.py --input_dir=/fsx/data/nlp/BERT/phase1/ --output_dir=/fsx/code/gradstats/BERT//results/pretrain_large_8/checkpoints --config_file=/fsx/code/gradstats/BERT/bert_base_config.json --bert_model=bert-large-uncased --train_batch_size=256 --max_seq_length=128 --max_predictions_per_seq=20 --max_steps=7038 --warmup_proportion=0.2843 --num_steps_per_checkpoint=8000 --learning_rate=6e-3 --seed=17 --fp16 --gradient_accumulation_steps=64 --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --do_train --json-summary /fsx/code/gradstats/BERT/results/pretrain_large_8/dllogger.json"
	#	ssh host $CMD
		rank=$((rank+1))
done 
