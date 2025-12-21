import torch
import torch.nn as nn
import numpy as np
import random
import os

# Blog: https://blog.csdn.net/fengbingchun/article/details/156129401

def set_seed(seed=42):
	os.environ['PYTHONHASHSEED'] = str(seed)
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

	random.seed(seed)
	np.random.seed(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)

	def seed_worker(worker_id):
		worker_seed = seed + worker_id
		np.random.seed(worker_seed)
		random.seed(worker_seed)

	return seed_worker

def test_random():
	data = [random.random() for _ in range(4)]
	print(f"random: {data}")

	data = [random.uniform(10, 20) for _ in range(4)]
	print(f"random: {data}")

def test_numpy():
	data = np.random.random(4)
	print(f"numpy: {data}")

	data = np.random.randn(4)
	print(f"numpy: {data}")

class TinyNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.fc = nn.Linear(8, 6)
		self.initialize_weights()

	def initialize_weights(self):
		nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')

	def forward(self, x):
		return self.fc(x)

def test_torch():
	data = torch.rand(4)
	print(f"torch: {data}")

	data = torch.randn(4)
	print(f"torch: {data}")

	model = TinyNet()
	weight_flat = model.fc.weight.flatten()
	for i in range(min(4, len(weight_flat))):
		print(f"{weight_flat[i]:.6f}", end=" ")
	print()

	# if num_workers is not 0 in DataLoader, then worker_init_fn and generator need to be set
	#	worker_init_fu = seed_worker # set_seed(seed)
	#	generator = torch.Generator().manual_seed(seed)

if __name__ == "__main__":
	seed_worker = set_seed(42) # seed_worker is used by DataLoader

	test_random()
	test_numpy()
	test_torch()

	print("====== execution completed ======")
