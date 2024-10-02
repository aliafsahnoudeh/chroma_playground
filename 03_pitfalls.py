import time

start = time.time()

from helper_utils import load_chroma, word_wrap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_collection = load_chroma(filename='./pdfs/Iran_History.pdf', collection_name='Iran_History', embedding_function=embedding_function)
chroma_collection.count()

import umap
import numpy as np
from tqdm import tqdm

embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings   

projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

import matplotlib.pyplot as plt

# plt.figure()
# plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10)
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('Projected Embeddings')
# plt.axis('off')
# plt.show()

# ------------------------------------------------------------------------------------------

# # Relevancy and Distraction
# query = "Who was cyrus the great?"

# results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])

# retrieved_documents = results['documents'][0]

# for document in results['documents'][0]:
#     print(word_wrap(document))
#     print('')
    
# query_embedding = embedding_function([query])[0]
# retrieved_embeddings = results['embeddings'][0]

# projected_query_embedding = project_embeddings([query_embedding], umap_transform)
# projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)


# Plot the projected query and retrieved documents in the embedding space
# plt.figure()
# plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
# plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker='X', color='r')
# plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')

# plt.gca().set_aspect('equal', 'datalim')
# plt.title(f'{query}')
# plt.axis('off')
# plt.show()


# ------------------------------------------------------------------------------------------

# query = "How Islam invaded Iran?"
# results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])

# retrieved_documents = results['documents'][0]

# for document in results['documents'][0]:
#     print(word_wrap(document))
#     print('')
    
    
# query_embedding = embedding_function([query])[0]
# retrieved_embeddings = results['embeddings'][0]

# projected_query_embedding = project_embeddings([query_embedding], umap_transform)
# projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)


# # Plot the projected query and retrieved documents in the embedding space
# plt.figure()
# plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
# plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker='X', color='r')
# plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')

# plt.gca().set_aspect('equal', 'datalim')
# plt.title(f'{query}')
# plt.axis('off')
# plt.show()


# ------------------------------------------------------------------------------------------

# query = "When exactly Iran was the most powerful country in the world?"
# results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])

# retrieved_documents = results['documents'][0]

# for document in results['documents'][0]:
#     print(word_wrap(document))
#     print('')
    
    
# query_embedding = embedding_function([query])[0]
# retrieved_embeddings = results['embeddings'][0]

# projected_query_embedding = project_embeddings([query_embedding], umap_transform)
# projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)


# # Plot the projected query and retrieved documents in the embedding space
# plt.figure()
# plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
# plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker='X', color='r')
# plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')

# plt.gca().set_aspect('equal', 'datalim')
# plt.title(f'{query}')
# plt.axis('off')

# ------------------------------------------------------------------------------------------

query = "Who was jimi hendrix and what he did?"
results = chroma_collection.query(query_texts=query, n_results=5, include=['documents', 'embeddings'])

retrieved_documents = results['documents'][0]

for document in results['documents'][0]:
    print(word_wrap(document))
    print('')

query_embedding = embedding_function([query])[0]
retrieved_embeddings = results['embeddings'][0]

projected_query_embedding = project_embeddings([query_embedding], umap_transform)
projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)


# Plot the projected query and retrieved documents in the embedding space
plt.figure()
plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color='gray')
plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker='X', color='r')
plt.scatter(projected_retrieved_embeddings[:, 0], projected_retrieved_embeddings[:, 1], s=100, facecolors='none', edgecolors='g')

plt.gca().set_aspect('equal', 'datalim')
plt.title(f'{query}')
plt.axis('off')
plt.show()
end = time.time()

print(end - start)