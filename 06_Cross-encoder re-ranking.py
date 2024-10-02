from helper_utils import load_chroma, word_wrap, project_embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np


embedding_function = SentenceTransformerEmbeddingFunction()

chroma_collection = load_chroma(filename='Iran_History.pdf', collection_name='Iran_History', embedding_function=embedding_function)
chroma_collection.count()

# Re-ranking the long tail
# instead of 5 results, we ask for 10 so we have a longer list to figure out the answer
query = "How many times was Iran invaded by a foreign force?"
results = chroma_collection.query(query_texts=query, n_results=10, include=['documents', 'embeddings'])

retrieved_documents = results['documents'][0]

for document in results['documents'][0]:
    print(word_wrap(document))
    print('')
    

from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [[query, doc] for doc in retrieved_documents]

scores = cross_encoder.predict(pairs)
print("Scores:")
for score in scores:
    print(score)
    
print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o+1)




# Re-ranking with Query Expansion
original_query = "What were the most important factors that contributed to increases in revenue?"
generated_queries = [
    "What were the major drivers of revenue growth?",
    "Were there any new product launches that contributed to the increase in revenue?",
    "Did any changes in pricing or promotions impact the revenue growth?",
    "What were the key market trends that facilitated the increase in revenue?",
    "Did any acquisitions or partnerships contribute to the revenue growth?"
]


queries = [original_query] + generated_queries

results = chroma_collection.query(query_texts=queries, n_results=10, include=['documents', 'embeddings'])
retrieved_documents = results['documents']

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

unique_documents = list(unique_documents)

pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])
    
scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)
    
print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)
    
