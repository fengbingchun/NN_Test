import colorama
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

# Blog: https://blog.csdn.net/fengbingchun/article/details/155393608

def main():
	tensor1 = torch.ones((4, 4), dtype=torch.int32)
	tensor2 = torch.zeros((8, 8), dtype=torch.int32)
	tensor2.fill_(2)
	name = "model.safetensors"

	tensors = {
		"weight1": tensor1,
		"weight2": tensor2
	}
	save_file(tensors, name)

	tensors1 = load_file(name)
	print(f"tensors1: {tensors1}")

	tensors2 = {}
	with safe_open(name, framework="pt", device="cpu") as f:
		for key in f.keys():
			print(f"key: {key}")
			tensors2[key] = f.get_tensor(key)
	print(f"tensors: {tensors2}")

if __name__ == "__main__":
	colorama.init(autoreset=True)

	main()

	print(colorama.Fore.GREEN + "====== execution completed ======")
