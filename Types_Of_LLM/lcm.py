import json, networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# 1) Concept inventory
concepts = ["refund policy","SLA","PCI compliance","PTO","on-call","RAG","vector database","Kubernetes"]
embed = SentenceTransformer("all-MiniLM-L6-v2")
C = embed.encode(concepts)

# 2) Encode a query and pick top concepts
def concepts_for(text, k=3):
    q = embed.encode([text])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(C)
    idx = nn.kneighbors(q, return_distance=False)[0]
    return [concepts[i] for i in idx]

# 3) Route generation using concepts (here we just print)
user = "How do we secure card data in our payment service?"
selected = concepts_for(user, k=2)
print("Concepts:", selected)

# 4) Optionally link to a knowledge graph
G = nx.Graph(); G.add_edges_from([("PCI compliance","tokenization"),("PCI compliance","encryption")])
neighbors = list(G.neighbors("PCI compliance"))
print("Related nodes:", neighbors)

# Downstream: feed concepts + KG neighbors into the prompt for grounded answers.