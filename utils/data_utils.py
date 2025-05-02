import torch
from transformers import AutoModelForCausalLM

class Adapter():
	def __init__(self, params):
		self.param_dict = params

	@classmethod
	def from_models(cls, base_model, tuned_model):
		param_dict = {}
		for (name, base_param), (n, tune_param) in zip(base_model.named_parameters(), tuned_model.named_parameters()):
			assert name == n, f"Parameter mismatch between {name} and {n}"
			param_dict[name] = tune_param - base_param

		return cls(param_dict)

	@classmethod
	def from_dict(cls, file_path):
		param_dict = torch.load(file_path, weights_only=True)
		return cls(param_dict)

	def apply(self, target_model, ratio=1.0):
		with torch.no_grad():
			for n, p in target_model.named_parameters():
				p += ratio * self.param_dict[n]

	def save(self, save_path):
		torch.save(self.param_dict, save_path)

def is_model_equal(m1, m2):
	for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
		assert n1 == n2, f"Name mismatch {n1} != {n2}"
		if torch.all(torch.isclose(p1, p2)):
			continue
		else:
			print(f"Parameter mismatch at {n1}, error of {(p1 - p2).sum()}")
			return False
	return True

