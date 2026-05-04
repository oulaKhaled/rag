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


# llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)

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
