from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM
from peft import LoraConfig

from utils.data_utils import Adapter, is_model_equal

def main():
	dataset = load_from_disk("./data/sara/statute_dataset/")
	model = AutoModelForCausalLM.from_pretrained("../models/llama3-8b")

	training_args = SFTConfig(
		max_length=2048,
		output_dir="/out/",
		learning_rate=1e-5,
		completion_only_loss=False,
		per_device_train_batch_size=1,
	)

	peft_args = LoraConfig(
		r=16,
		lora_alpha=32,
		lora_dropout=0.03,
		target_modules="all-linear",
		modules_to_save=["lm_head", "embed_token"],
		task_type="CAUSAL_LM",
	)

	trainer = SFTTrainer(
		model,
		train_dataset=dataset,
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
	create_reasoning_adapter()


