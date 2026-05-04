import os
from dotenv import load_dotenv
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.messages import SystemMessage, HumanMessage
import google.generativeai as genai
from chromadb.utils import embedding_functions

# from google import genai

load_dotenv()
api_key = os.getenv("API_KEY")
embeddings = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    # model="models/gemini-embedding-001",
    api_key=api_key,
)


# result = embeddings.embed_query("hello world")
# print(result[:5])


# initialize the Chroma client with presistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=embeddings
)


# ## create client
# genai.configure(api_key=api_key)
# client = genai.GenerativeModel("gemini-flash-latest")

# # client
# response = client.generate_content("How does AI work")
# print(response.text)


llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)

# messages = [
#     SystemMessage(content="You are a good therapist"),
#     HumanMessage(content="how can I feel better?"),
# ]

# response = llm.invoke(messages)
# print(response.content)


def load_documents_from_directory(directory_path):
    documnets = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documnets.append({"id": filename, "text": file.read()})

    return documnets


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# print(f"Split documents into {len(chunked_documents)} chunks")


# Function to generate embeddings using OpenAI API
def get_gemini_embedding(text):
    response = embeddings.embed_query(text)
    print("==== Generating embeddings... ====")
    return response


# Generate embeddings for the document chunks
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    doc["embedding"] = get_gemini_embedding(doc["text"])

# print(doc["embedding"])

# Upsert documents with embeddings into Chroma
for doc in chunked_documents:
    print("==== Inserting chunks into db;;; ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    # query_embedding = get_openai_embedding(question)
    results = collection.query(query_texts=question, n_results=n_results)

    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks
    # for idx, document in enumerate(results["documents"][0]):
    #     doc_id = results["ids"][0][idx]
    #     distance = results["distances"][0][idx]
    #     print(f"Found document chunk: {document} (ID: {doc_id}, Distance: {distance})")


# Function to generate a response from OpenAI
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)

    messages = [
        SystemMessage(
            content=(
                "You are an assistant for question-answering tasks. Use the following pieces of "
                "retrieved context to answer the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the answer concise."
                "\n\nContext:\n" + context
            )
        ),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    return response.content


# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)
