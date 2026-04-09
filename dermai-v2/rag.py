import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DermAIRAG:
    def __init__(self, guidelines_path="guidelines.md"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus = []
        self._load_corpus(guidelines_path)
        
        # Build FAISS index
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        
        if self.corpus:
            embeddings = self.model.encode(self.corpus)
            self.index.add(np.array(embeddings).astype('float32'))

    def _load_corpus(self, path):
        if not os.path.exists(path):
            self.corpus = ["NCCN Guidelines for Melanoma not found. Please ensure guidelines.md exists."]
            return
            
        with open(path, 'r') as f:
            content = f.read()
            # Split by double newline to get paragraphs/sections
            self.corpus = [p.strip() for p in content.split('\n\n') if p.strip()]

    def search(self, query, top_k=2):
        if not self.corpus:
            return "No information available."
            
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for i in indices[0]:
            if i < len(self.corpus):
                results.append(self.corpus[i])
                
        return "\n\n---\n\n".join(results)

def get_rag_concierge():
    # Use absolute path if necessary
    base_dir = os.path.dirname(__file__)
    guidelines_path = os.path.join(base_dir, "guidelines.md")
    return DermAIRAG(guidelines_path)
