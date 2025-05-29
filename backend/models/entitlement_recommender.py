import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import requests

class EntitlementRecommender:
    def __init__(self, csv_path, model_name='all-MiniLM-L6-v2', embeddings_cache='embeddings.pkl'):
        self.csv_path = csv_path
        self.embeddings_cache = embeddings_cache
        self.df = None
        self.embeddings = None
        self.model = SentenceTransformer(model_name)

        self.load_data()
        self.load_or_create_embeddings()
    
    def load_data(self):
        """Load entitlements data from CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            # required_columns = ['Entitlements', 'Description', 'Tags']
            
            # if not all(col in self.df.columns for col in required_columns):
            #     raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # self.df = self.df.dropna(subset=required_columns)
            # self.df['combined_text'] = (
            #     self.df['Entitlements'] + ' ' + 
            #     self.df['Description'] + ' ' + 
            #     self.df['Tags']
            # )
            
            # print(f"Loaded {len(self.df)} entitlements from {self.csv_path}")
            
            required_columns = ['Entitlements', 'Description', 'Tags']
            optional_columns = ['Dependency']

            all_required = required_columns + [col for col in optional_columns if col in self.df.columns]

            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")

            self.df = self.df.dropna(subset=required_columns)

            if 'Dependency' in self.df.columns:
                self.df['Dependency'] = self.df['Dependency'].fillna('')

            self.df['combined_text'] = (
                self.df['Entitlements'] + ' ' +
                self.df['Description'] + ' ' +
                self.df['Tags']
            )
            print(f"Loaded {len(self.df)} entitlements from {self.csv_path}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def load_or_create_embeddings(self):
        """Load cached embeddings or create new ones"""
        if os.path.exists(self.embeddings_cache):
            try:
                with open(self.embeddings_cache, 'rb') as f:
                    cache_data = pickle.load(f)
                    

                if (len(cache_data['embeddings']) == len(self.df) and 
                    cache_data['csv_modified'] == os.path.getmtime(self.csv_path)):
                    self.embeddings = cache_data['embeddings']
                    print("Loaded cached embeddings")
                    return
            except Exception as e:
                print(f"Error loading cached embeddings: {e}")
        
        print("Creating embeddings... This may take a moment.")
        self.embeddings = self.model.encode(self.df['combined_text'].tolist())
        
        try:
            cache_data = {
                'embeddings': self.embeddings,
                'csv_modified': os.path.getmtime(self.csv_path)
            }
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(cache_data, f)
            print("Embeddings cached for future use")
        except Exception as e:
            print(f"Warning: Could not cache embeddings: {e}")
    
    def search_entitlements(self, query, top_k=10, similarity_threshold=0.3):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
    
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            score = similarities[idx]
            if score >= similarity_threshold:
                results.append({
                    'entitlement': self.df.iloc[idx]['Entitlements'],
                    'description': self.df.iloc[idx]['Description'],
                    'tags': self.df.iloc[idx]['Tags'],
                    'dependency': self.df.iloc[idx].get('Dependency', ''),
                    'similarity_score': float(score),
                    'relevance': self._get_relevance_label(score)
                })
        return results
    
    def _get_relevance_label(self, score):
        if score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def format_results(self, results, max_results=None):
        if not results:
            return "No relevant entitlements found. Try rephrasing your query or being more specific."
        
        if max_results:
            results = results[:max_results]
        
        output = f"\n{'='*60}\n"
        output += f"Found {len(results)} relevant entitlement(s):\n"
        output += f"{'='*60}\n"
        
        for i, result in enumerate(results, 1):
            if result['relevance'] != "Low":
                output += f"\n{i}. {result['entitlement']}\n"
                output += f"   Relevance: {result['relevance']} ({result['similarity_score']:.3f})\n"
                output += f"   Description: {result['description']}\n"
                output += f"   Tags: {result['tags']}\n"
                if result['dependency']:
                    output += f"   Depends on: {result['dependency']}\n"
                output += f"   {'-'*50}\n"

        return output

class LlamaIntegration:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.available = self._check_ollama_availability()
    
    def _check_ollama_availability(self):
        """Check if Ollama is running locally"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def enhance_recommendations(self, query, recommendations):
        if not self.available:
            return None
        
        try:
            prompt = f"""Based on the user query: "{query}"

The following entitlements were found:
{chr(10).join([f"- {r['entitlement']}: {r['description']}" for r in recommendations[:5]])}

Provide a brief, helpful explanation of why these entitlements might be relevant to the user's needs. Be concise and practical."""

            payload = {
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200
                }
            }
            
            response = requests.post(f"{self.base_url}/api/generate", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except Exception as e:
            print(f"Warning: Could not get Llama enhancement: {e}")
        
        return None
