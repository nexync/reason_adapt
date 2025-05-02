import os
from datasets import *

def main():
	folder_path = "./statutes/source/"
	res = []

	for filename in os.listdir(folder_path):
		if filename[0] == ".":
			continue
		file_path = os.path.join(folder_path, filename)
		with open(file_path, "r", encoding="utf-8") as f:
			text = f.read()
			res.append(text)

	dataset = Dataset.from_dict({"text": res})
	dataset.save_to_disk("./")

if __name__ == "__main__":
	main()
