from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM
from peft import LoraConfig

from utils.data_utils import Adapter, is_model_equal

def main():
	dataset_path = "./data/lichess/chess_game_dataset"
	model_path = "../models/llama3-8b"

	dataset = load_from_disk(dataset_path)
	model = AutoModelForCausalLM.from_pretrained(model_path)

	training_args = SFTConfig(
		max_length=2048,
		output_dir="./chess_checkpoints/all_rank_16_alpha_16/",
		learning_rate=1e-5,
		completion_only_loss=False,
		per_device_train_batch_size=1,
		num_train_epochs=1,
	)

	peft_args = LoraConfig(
		r=16,
		lora_alpha=16,
		lora_dropout=0.05,
		target_modules="all-linear",
		modules_to_save=["lm_head", "embed_token"],
		task_type="CAUSAL_LM",
	)

	trainer = SFTTrainer(
		model,
		train_dataset=dataset.select(list(range(1000))),
		args=training_args,
		peft_config=peft_args,
	)

	trainer.train()

def create_reasoning_adapter():
	base = AutoModelForCausalLM.from_pretrained("../models/llama3-8b")
	tune = AutoModelForCausalLM.from_pretrained("../models/llama3-8b-r1")

	adapt = Adapter.from_models(base, tune)
	adapt.save("./out/adapter.pt")

	
if __name__ == "__main__":
	main()


