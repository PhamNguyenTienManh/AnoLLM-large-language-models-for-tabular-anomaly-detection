import os
import sys
from pathlib import Path
import argparse

from ssl_patch import patch_ssl_context

patch_ssl_context()

import numpy as np
import torch
import time

import torch.distributed as dist

from src.data_utils import load_data, DATA_MAP, get_text_columns, get_max_length_dict
from train_anollm import get_run_name


def is_distributed():
	return dist.is_available() and dist.is_initialized()


def get_rank():
	return dist.get_rank() if is_distributed() else 0


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type = str, default='wine', choices = [d.lower() for d in DATA_MAP.keys()],
					help="Name of datasets in the ODDS benchmark")
	parser.add_argument("--exp_dir", type = str, default=None)
	parser.add_argument("--setting", type = str, default='semi_supervised', choices = ['semi_supervised', 'unsupervised'], help="semi_supervised:an uncontaminated, unsupervised setting; unsupervised:a contaminated, unsupervised setting")
	
	#dataset hyperparameters
	parser.add_argument("--data_dir", type = str, default='data')
	parser.add_argument("--n_splits", type = int, default=5)
	parser.add_argument("--split_idx", type = int, default=None) # 0 to n_split-1
	parser.add_argument("--train_ratio", type = float, default=0.5)
	parser.add_argument("--seed", type = int, default=42)
	# binning
	parser.add_argument("--binning", type = str, choices=['quantile', 'equal_width', 'language', 'none', 'standard'], default='standard')
	parser.add_argument("--n_buckets", type = int, default=10)
	parser.add_argument("--remove_feature_name", action = 'store_true')
	
	# model hyperparameters (for getting the model name)
	parser.add_argument("--model", type = str, choices = ['gpt2', 'distilgpt2', 'smol', 'smol-360', 'smol-1.7b'], default='smol')
	parser.add_argument("--lora", action='store_true', default=False)
	parser.add_argument("--lr", type = float, default=5e-5)
	parser.add_argument("--random_init", action='store_true', default=False)
	parser.add_argument("--no_random_permutation", action='store_true', default=False)
	
	#testing
	parser.add_argument("--batch_size", type = int, default=128) # per gpu
	parser.add_argument("--n_permutations", type = int, default=100) # per gpu
	args = parser.parse_args()
	
	if args.model == 'smol':
		args.model = 'HuggingFaceTB/SmolLM-135M'
	elif args.model == 'smol-360':
		args.model = 'HuggingFaceTB/SmolLM-360M'
	elif args.model == 'smol-1.7b':	
		args.model = 'HuggingFaceTB/SmolLM-1.7B'
	
	return args

def main():
	# Set CUDA devices for each process
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	world_size = dist.get_world_size() if is_distributed() else 1
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
	if use_cuda:
		torch.cuda.set_device(local_rank)
	
	args = get_args()

	if args.exp_dir is None:
		args.exp_dir = Path('exp') / args.dataset / args.setting / "split{}".format(args.n_splits) / "split{}".format(args.split_idx)
	else:
		args.exp_dir = Path(args.exp_dir)
	
	if not os.path.exists(args.exp_dir):
		raise ValueError("Experiment directory {} does not exist".format(args.exp_dir))
		
	score_dir = args.exp_dir / 'scores'
	run_name = get_run_name(args)

	score_path = score_dir / "{}.npy".format(run_name)
	print("score_path:",  score_path)	
	if get_rank() == 0:
		os.makedirs(score_dir, exist_ok = True)

	remainder = args.n_permutations % world_size
	
	X_train, X_test, y_train, y_test = load_data(args)
	
	if not os.path.exists(score_path):
		model_dir = args.exp_dir / 'models'
		model_path = model_dir / '{}.pt'.format(run_name)
		
		efficient_finetuning = 'lora' if args.lora else ''
		max_length_dict = get_max_length_dict(args.dataset)
		text_columns = get_text_columns(args.dataset)
		from anollm import AnoLLM

		model = AnoLLM(args.model,
						efficient_finetuning = efficient_finetuning,
						model_path = model_path,
						max_length_dict=max_length_dict, 
						textual_columns = text_columns,
						no_random_permutation=args.no_random_permutation,
						bf16=use_cuda,
				)
		print(text_columns, max_length_dict)
		
		model.load_from_state_dict(model_path)
		model.model.to(device)
			
		# Move the model to the appropriate GPU
		# Wrap the model for distributed training
		if is_distributed():
			if use_cuda:
				model.model = torch.nn.parallel.DistributedDataParallel(
					model.model, device_ids=[local_rank], output_device=local_rank
				)
			else:
				model.model = torch.nn.parallel.DistributedDataParallel(model.model)
		n_perm = int(args.n_permutations / world_size) 
		n_perm = n_perm + 1 if local_rank < remainder else n_perm

		start_time = time.time()	
		scores = model.decision_function(X_test, 
										n_permutations = n_perm, 
										batch_size = args.batch_size, 
										device = str(device),
		)
		end_time = time.time()

		if is_distributed():
			all_scores = [None for _ in range(world_size)]
			dist.all_gather_object(all_scores, scores)
		else:
			all_scores = [scores]

		if get_rank() == 0:
			
			print("Inference time:", end_time - start_time)
			
			run_time_dir = args.exp_dir / "run_time" / "test"
			os.makedirs(run_time_dir, exist_ok = True)
			run_time_path = run_time_dir / "{}.txt".format(run_name)
			with open(run_time_path, 'w') as f:
				f.write(str(end_time - start_time))
			
			all_scores = np.concatenate(all_scores, axis = 1)
			mean_scores = np.mean(scores, axis = 1)
			np.save(score_path, mean_scores)
			raw_score_path =  score_dir / "raw_{}.npy".format(run_name) 
			np.save(raw_score_path, all_scores)
	
	if is_distributed():
		dist.destroy_process_group()
	
if __name__ == '__main__':
	if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
		get_args()
		sys.exit(0)

	if {"RANK", "WORLD_SIZE", "LOCAL_RANK"}.issubset(os.environ):
		backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
		dist.init_process_group(backend=backend)
	main()
