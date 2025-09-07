import argparse
import colorama
import ollama

# Blog: https://blog.csdn.net/fengbingchun/article/details/151286661

def parse_args():
	parser = argparse.ArgumentParser(description="ollama chat")
	parser.add_argument("--task", required=True, type=str, choices=["model_list", "generate", "chat"], help="specify what kind of task")
	parser.add_argument("--model", type=str, help="model name, for example:deepseek-r1:1.5b")
	parser.add_argument("--prompt", type=str, default="", help="system prompt")

	args = parser.parse_args()
	return args

def model_list():
	response = ollama.list()
	for model in response.models:
		print(f"name: {model.model}")
		print(f"\tsize(MB): {(model.size.real / 1024 / 1024):.2f}")

		if model.details:
			print(f"\tformat: {model.details.format}")
			print(f"\tfamily: {model.details.family}")
			print(f"\tparameter size: {model.details.parameter_size}")
			print(f"\tquantization level: {model.details.quantization_level}")

def generate(model, prompt):
	try:
		stream = ollama.generate(model=model, prompt=prompt, stream=True)
		print("AI: ", end="", flush=True)

		for chunk in stream:
			if 'response' in chunk:
				content = chunk['response']
				print(content, end="", flush=True)
		print() # line break
	except Exception as e:
		print(f"Error: {e}")

def chat(model, system_prompt):
	if system_prompt != "":
		messages = [{'role': 'system', 'content': system_prompt}]
	else:
		messages = []

	while True:
		user_input = input("\nYou: ").strip()

		if user_input.lower() in ['quit', 'exit', 'q']:
			break

		if not user_input: # empty input
			continue

		messages.append({'role': 'user', 'content': user_input})

		try:
			stream = ollama.chat(model=model, messages=messages, stream=True)
			print("AI: ", end="", flush=True)

			assistant_reply = ""
			for chunk in stream:
				if 'message' in chunk and 'content' in chunk['message']:
					content = chunk['message']['content']
					print(content, end="", flush=True)
					assistant_reply += content

			print() # line break

			messages.append({'role': 'assistant', 'content': assistant_reply})
		except Exception as e:
			print(f"Error: {e}")
			if messages[-1]['role'] == 'user':
				messages.pop()

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "model_list":
		model_list()
	elif args.task == "generate":
		generate(args.model, args.prompt)
	elif args.task == "chat":
		chat(args.model, args.prompt)

	print(colorama.Fore.GREEN + "====== execution completed ======")
