from helper_utils import word_wrap
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


# reading the pdf file and splitting pages
reader = PdfReader("./pdfs/Persian-Literature.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]

print(word_wrap(pdf_texts[0]))


# splitting the text by certain characters into chunks
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0
)
character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

print(word_wrap(character_split_texts[10]))
print(f"\nTotal chunks: {len(character_split_texts)}")


# splitting the chunks from previous step by the number of tokens
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

print(word_wrap(token_split_texts[10]))
print(f"\nTotal chunks: {len(token_split_texts)}")

# initializing the embedding function, which is created using the SentenceTransformer library (SBERT)
embedding_function = SentenceTransformerEmbeddingFunction()
print(embedding_function([token_split_texts[10]]))

# passing the embedding function to the ChromaDB client and creating a collection
# chroma will use the same embedding function to code the query
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("Persian-Literature", embedding_function=embedding_function)

# index of each document will the id of the chunk in chroma collection
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()


query = "What was the seven labors of Rustem?"

# fetch the relevant documents from the collection, maximum 5
results = chroma_collection.query(query_texts=[query], n_results=5)
# results for the first query (we have only one here!)
retrieved_documents = results['documents'][0]

for document in retrieved_documents:
    print(word_wrap(document))
    print('\n')

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

openai_client = OpenAI()


# making gpt from a modle that remebers facts into a model that processes information
def rag(query, retrieved_documents, model="gpt-4"):
    information = "\n\n".join(retrieved_documents)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful persian litrature research assistant. Your users are asking questions about information contained in a book."
            "You will be shown the user's question, and the relevant information from the book. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


output = rag(query=query, retrieved_documents=retrieved_documents)

print(word_wrap(output))
