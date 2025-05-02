import os
import pickle
import re
import sys

def get_answer(response_dict):
	gold = float(response_dict["label"])
	match = re.search(r"\\boxed{(\d+)}", response_dict["response"])

	if match:
		pred = float(match.group(1))
		if (abs(pred - gold) / (gold+1e-5) ) < 0.01:
			return 1
	return 0

def main():
	assert len(sys.argv) == 2, "Missing directory of results to collate"
	
	correct, total = 0, 0

	file_dir = sys.argv[1]
	for file in os.listdir(file_dir):
		file_path = os.path.join(file_dir, file)
		with open(file_path, "rb") as f:
			res = pickle.load(f)
		
		for d in res:
			correct += get_answer(d)
			total += 1

	print(f"Accuracy of this batch is {correct}/{total}")

if __name__ == "__main__":
	main()
