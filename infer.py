import argparse
import yaml
import torch
import tqdm
import time
import pickle 
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils.infer_utils import load_statutes, load_cases, compose_prompt
from utils.data_utils import Adapter

def parse_args():
	parent_parser = argparse.ArgumentParser(add_help=False)

	parser = argparse.ArgumentParser(parents=[parent_parser])
	cfg_parser = argparse.ArgumentParser(parents=[parent_parser])

	cfg_parser.add_argument("--config", default="default")
	cfg_parser.add_argument("--config_file", required=True)
	config_args, _ = cfg_parser.parse_known_args()

	with open(config_args.config_file) as f:
		config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]

	parser.add_argument("--use_tuned", type=bool)

	# If we only need the tuned model (r1/llama)
	parser.add_argument("--tune_model_path", type=str)	

	# If we need to load an re-adapt model
	parser.add_argument("--base_model_path", type=str)
	parser.add_argument("--checkpoint_path", type=str)
	parser.add_argument("--adapter_path", type=str)
	parser.add_argument("--lam_ratio", type=float)

	parser.add_argument("--tokenizer_path", type=str)
	parser.add_argument("--dataset_path", type=str)

	parser.add_argument("--use_cuda", type=bool)
	parser.add_argument("--batch_size", type=int)

	parser.add_argument("--start_index", type=int, default=0)
	parser.add_argument("--stop_index", type=int, default=100)

	parser.set_defaults(**config)
	args, _ = parser.parse_known_args()

	return args
	

def main():
	args = parse_args()
	
	# Model setup
	if args.naive:
		assert args.tune_model_path is not None
		model = AutoModelForCausalLM.from_pretrained(args.tune_model_path)
	else:
		for val in [args.base_model_path, args.adapter_path, args.lam_ratio, args.checkpoint_path]:
			assert val is not None
		base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
		reasoning_adapter = Adapter.from_dict(args.adapter_path)
		reasoning_adapter.apply(base_model, args.lam_ratio)
		model = PeftModel.from_pretrained(base_model, args.checkpoint_path)

	if args.use_cuda and torch.cuda.is_available():
		print("Using cuda")
		model.to("cuda")
	model.eval()

	# Data setup
	cases = load_cases(args.dataset_path+"/cases/")
	statutes = load_statutes(args.dataset_path+"/statutes/source/")

	cases = cases[args.start_index:args.stop_index]
	print(f"Evaluating from {args.start_index} to {args.stop_index}")

	
	save_dir = f"{args.save_dir}/naive_{args.naive}/"
	save_file = f"res_{args.start_index}_{args.stop_index}.pkl"
	save_path = os.path.join(save_dir, save_file)
	os.makedirs(save_dir, exist_ok=True)

	
	# Tokenizer setup
	tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
	tok.add_bos_token = False
	special_tokens_dict = {
		"bos": tok.decode(128000),
		"eos": tok.decode(128001),
		"user": tok.decode(128011),
		"assistant": tok.decode(128012),
		"bot": tok.decode(128013),
		"eot": tok.decode(128014),
		"prompt": "<<PROMPT>>",
	}


	# Run inference
	input_batch = []
	responses = []
	for i, case in tqdm.tqdm(enumerate(cases), desc="Evaluating"):
		prompt = compose_prompt(statutes, case)
		
		model_prompt = "".join([special_tokens_dict[val] for val in ["bos", "user", "prompt", "assistant", "bot"]]) + "\n"
		model_prompt = model_prompt.replace("<<PROMPT>>", prompt)
		input_batch.append(model_prompt)

		# For storing answers
		case["prompt"] = model_prompt
		responses.append(case)

		if len(input_batch) == args.batch_size or i+1 == len(cases):
			model_inputs = tok(input_batch, return_tensors="pt", padding=True).to(model.device)
			
			with torch.no_grad():
				out = model.generate(
					inputs=model_inputs.input_ids,
					attention_mask=model_inputs.attention_mask,
					max_new_tokens=16000,
					do_sample=True,
					temperature=0.6,
					repetition_penalty=1.0,
					eos_token_id=128001,
					pad_token_id=128015,
				)

			# Update responses			
			for j in range(len(input_batch)):
				curr_index = i-len(input_batch)+j+1
				assert input_batch[j] == responses[curr_index]["prompt"]

				try:
					pad_index = torch.nonzero(out[j] == 128015)[0][0]
				except IndexError:
					pad_index = len(out[j])

				responses[curr_index]["response"] = tok.decode(out[j][:pad_index])

			# Clear input batch
			input_batch = []

			# Intermediate saving
			with open(save_path, "wb") as f:
				pickle.dump(responses, f)
	
	# Save to file
	with open(save_path, "wb") as f:
		pickle.dump(responses, f)
	
	print("Finished evaluation")
			
if __name__ == "__main__":
	main()
