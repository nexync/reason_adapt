from openai import OpenAI
from datasets import load_from_disk

def main():
	dataset = load_from_disk("./data/sara/statute_dataset/")
	
	instructions = "You are a tax expert that will provide clear and concise paraphrases to the tax statutes given to you. You are not to omit or add any additional information, just paraphrase the information in the given statute in a concise manner. Ensure that you retain the same structure as the given statute in the paraphrased output."

	prompt = "Please paraphrases the following statute. Do not alter the line headings; only paraphrase the textual content.\n\n"

	res = []
	client = OpenAI()
	for item in dataset:
		statute = item["text"]
		response = client.responses.create(
			model ="gpt-4.1",
			instructions=instructions,
			input=prompt+statute,
		)

		res.append(response.output_text)

	for para in res:
		dataset = dataset.add_item({"text": para})

	dataset.save_to_disk("./data/sara/paraphrased_datset/")

if __name__ == "__main__":
	main()
