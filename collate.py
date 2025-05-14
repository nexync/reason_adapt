import os
import pickle
import re
import sys
from transformers import AutoTokenizer

def get_answer(response_dict, constraint="hard"):
	gold = float(response_dict["label"])

	if constraint == "hard":
		match = re.search(r"\\boxed{([\d,.]+)}", response_dict["response"])
	elif constraint == "soft":
		match = re.search(r"([\d,.]+)(?!.*[\d]+)", response_dict["response"])
	else:
		raise Exception

	if match:
		pred = float(match.group(1).replace(",",""))
		if (abs(pred - gold) / (gold+1e-5) ) < 0.01:
			return 1
	return 0

def check_eos(response_dict, tok):
	if tok.decode(128001) in response_dict["response"][-1000:]:
		return 1
	return 0

def main():
	assert len(sys.argv) == 2, "Missing directory of results to collate"
	tok = AutoTokenizer.from_pretrained("/scratch/bvandur1/jeffc/models/llama3-8b-r1")

	correct, total = 0, 0

	file_dir = sys.argv[1]
	for file in os.listdir(file_dir):
		file_path = os.path.join(file_dir, file)
		with open(file_path, "rb") as f:
			res = pickle.load(f)
		
		for d in res:
			correct += get_answer(d, constraint="hard")
			total += 1

	print(f"Accuracy of this batch is {correct}/{total}")

if __name__ == "__main__":
	main()
