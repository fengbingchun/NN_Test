import argparse
import colorama
from pathlib import Path
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Blog: https://blog.csdn.net/fengbingchun/article/details/153878903

def parse_args():
	parser = argparse.ArgumentParser(description="rag qa chat: langchain")
	parser.add_argument("--llm_model", type=str, default="qwen3:1.7b", help="llm model name")
	parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-zh-v1.5", help="Embedding model name")
	parser.add_argument("--jsonl", type=str, default="csdn.jsonl", help="jsonl(JSON Lines) file")
	parser.add_argument("--db_dir", type=str, default="chroma_db_langchain", help="vector database(chromadb) storage path")
	parser.add_argument("--similarity_threshold", type=float, default=0.8, help="similarity threshold, the higher the stricter")
	parser.add_argument("--question", type=str, required=True, help="question text")

	args = parser.parse_args()
	return args

def load_embed_model(model_name):
	return HuggingFaceEmbeddings(
		model_name=model_name,
		model_kwargs={"device": "cpu"},
		encode_kwargs={"normalize_embeddings": True}
	)

def load_jsonl_docs(jsonl):
	docs = []
	with open(jsonl, "r", encoding="utf-8") as f:
		for line in f:
			try:
				data = json.loads(line.strip())
				doc = Document(
					page_content = data.get("question", ""),
					metadata = {"answer": data.get("answer", "")}
				)
				docs.append(doc)
			except json.JSONDecodeError as e:
				print(colorama.Fore.RED + f"Error parsing line: {e}")
				continue

	print(f"number of loaded data items: {len(docs)}")
	return docs

def build_vector_db(db_dir, jsonl, embed_model):
	db_path = Path(db_dir)
	if db_path.exists() and any(db_path.glob("*")):
		return Chroma(persist_directory=db_dir, embedding_function=embed_model, collection_name="csdn_qa_langchain")
	else:
		docs = load_jsonl_docs(jsonl)
		return Chroma.from_documents(documents=docs, embedding=embed_model, persist_directory=db_dir, collection_name="csdn_qa_langchain")

def retrieve_answer(vectorstore, similarity_threshold, question):
	results = vectorstore.similarity_search_with_score(question, k=1)
	if not results:
		return None

	doc, score = results[0]
	similarity = 1 - score
	print(f"similarity: {similarity:.4f}")

	if similarity >= similarity_threshold:
		return f"question: {doc.page_content}; link: {doc.metadata}"
	return None

def chat(llm_model_name, embed_model_name, jsonl, db_dir, similarity_threshold, question):
	embed_model = load_embed_model(embed_model_name)

	vectorstore = build_vector_db(db_dir, jsonl, embed_model)
	ans = retrieve_answer(vectorstore, similarity_threshold, question)
	if ans:
		print(ans)
	else:
		try:
			chat = ChatOllama(model=llm_model_name, streaming=True)
			print("Answer: ", end="", flush=True)
			for chunk in chat.stream([HumanMessage(content=question)]):
				print(chunk.content, end="", flush=True)
			print() # line break
		except Exception as e:
			print(f"Error: {e}")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	chat(args.llm_model, args.embed_model, args.jsonl, args.db_dir, args.similarity_threshold, args.question)

	print(colorama.Fore.GREEN + "====== execution completed ======")
