import argparse
import colorama
import ollama
import json
from tqdm import tqdm
import chromadb
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Blog: https://blog.csdn.net/fengbingchun/article/details/153526949

def parse_args():
	parser = argparse.ArgumentParser(description="ollama rag chat: sentence-transformers")
	parser.add_argument("--llm_model", type=str, default="qwen3:1.7b", help="llm model name, for example:qwen3:1.7b")
	parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-zh-v1.5", help="sentence-transformers model")
	parser.add_argument("--jsonl", type=str, default="csdn.jsonl", help="jsonl(JSON Lines) file")
	parser.add_argument("--db_dir", type=str, default="chroma_db_sentence_transformers", help="vector database(chromadb) storage path")
	parser.add_argument("--similarity_threshold", type=float, default=0.8, help="similarity threshold, the higher the stricter")
	parser.add_argument("--question", type=str, help="question")

	args = parser.parse_args()
	return args

def load_embed_model(model_name):
	return SentenceTransformer(model_name) # if the model already exists locally, add the parameter: local_files_only=True

def build_vector_db(db_dir, jsonl, embed_model):
	client = chromadb.PersistentClient(path=db_dir)
	collection = client.get_or_create_collection(name="csdn_qa_sentence-transformers", metadata={"hnsw:space":"cosine"})
	if collection.count() > 0:
		return collection

	print("start building vector database ...")

	with open(jsonl, "r", encoding="utf-8") as f:
		data = [json.loads(line.strip()) for line in f]

	for i, item in enumerate(tqdm(data, desc="Embedding with sentence-transformers")):
		question = item["question"]
		answer = item["answer"]

		emb = embed_model.encode(question, normalize_embeddings=True).tolist()

		collection.add(
			ids=[str(i)],
			embeddings=[emb],
			metadatas=[{"question": question, "answer": answer}]
		)
		time.sleep(0.05)

	print(f"vector database is built and a total of {len(data)} entries are imported")
	return collection

def retrieve_answer(embed_model, collection, similarity_threshold, question):
	query_vec = embed_model.encode(question, normalize_embeddings=True).tolist()
	results = collection.query(query_embeddings=[query_vec], n_results=1)
	print(f"vector len: {len(query_vec)}; norm: {np.linalg.norm(query_vec):.4f}; vector: {query_vec[:5]}")
	if not results["ids"]:
		return None

	meta = results["metadatas"][0][0]
	score = 1 - results["distances"][0][0]
	print(f"similarity: {score:.4f}")
	if score >= similarity_threshold:
		return f"question: {meta['question']}; link: {meta['answer']}"
	return None

def chat(llm_model_name, embed_model_name, jsonl, db_dir, similarity_threshold, question):
	embed_model = load_embed_model(embed_model_name)

	collection = build_vector_db(db_dir, jsonl, embed_model)
	ans = retrieve_answer(embed_model, collection, similarity_threshold, question)
	if ans:
		print(ans)
	else:
		try:
			stream = ollama.chat(model=llm_model_name, messages=[{"role": "user", "content": question}], stream=True)
			print("Answer: ", end="", flush=True)

			for chunk in stream:
				if 'message' in chunk and 'content' in chunk['message']:
					content = chunk['message']['content']
					print(content, end="", flush=True)
			print() # line break
		except Exception as e:
			print(f"Error: {e}")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	chat(args.llm_model, args.embed_model, args.jsonl, args.db_dir, args.similarity_threshold, args.question)

	print(colorama.Fore.GREEN + "====== execution completed ======")
