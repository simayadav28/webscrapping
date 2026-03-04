from ollama import Client
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# client = Client(host="http://localhost:11434")

client = chromadb.PersistentClient()
remote_client = Client(host="http://172.16.10.171:11434")
collection = client.get_or_create_collection(name="articles_demo")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

# Safe counter loading
try:
    with open("counter.txt", "r", encoding="utf-8") as f:
        counter = int(f.read().strip())
except:
    counter = 0

print("Reading article.json and generating embeddings...")

with open("art.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, article in enumerate(data):
        content = article["summary"]

        sentences = text_splitter.split_text(content)

        for each_sentence in sentences:
            print(f"search_document: {each_sentence}")
            embedding = remote_client.embed(model="nomic-embed-text", input=f"search_document: {each_sentence}")
            print(embedding)
            embedding = embedding["embeddings"][0]
            collection.add(
                ids=[f"article_{counter}"],
                embeddings=[embedding],
                documents=[each_sentence],
                metadatas=[{"title": article["title"]}],
            )

            counter += 1

print("Database built successfully!")

# Save counter
with open("counter.txt", "w", encoding="utf-8") as f:
    f.write(str(counter))


    
# Query
#query = "are there any predicted hindrance for upcoming election ?"

while True:
    print("-------------------------------------------------")
    query = input("🤖How may I help you? \n ")

    if query== "exit":
        break
    
    query_embed = remote_client.embed(
        model="nomic-embed-text",
        input=f"query: {query}"
    )["embeddings"][0]

    results = collection.query(query_embeddings=[query_embed], n_results=1)

    print("\nQuestion:", query)
    print("\nTitle:", results["metadatas"][0][0]["title"])
    print(results["documents"][0][0])

    context='\n'.join(results["documents"][0])

    prompt=f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"

    Context: {context}

    Question: {query}

    Answer:"""

    # prompt = "Are there any predicted hindrances for the upcoming election?"

    response = remote_client.generate(
        model="freehuntx/qwen3-coder:14b",
        prompt=prompt,
        options={
            "num_predict": 300,
            "temperature": 0
        }
    )

    print(response["response"])