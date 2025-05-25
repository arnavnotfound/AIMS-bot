#!/usr/bin/env python3
"""
Entitlement Recommendation System
A CLI-based chatbot that recommends entitlements based on user queries about their department, work, and projects.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import argparse
import json
from datetime import datetime
import requests
import time

class EntitlementRecommender:
    def __init__(self, csv_path, model_name='all-MiniLM-L6-v2', embeddings_cache='embeddings.pkl'):
        """
        Initialize the entitlement recommender
        
        Args:
            csv_path: Path to CSV file with entitlements
            model_name: Sentence transformer model name
            embeddings_cache: Path to cache embeddings
        """
        self.csv_path = csv_path
        self.embeddings_cache = embeddings_cache
        self.df = None
        self.embeddings = None
        self.model = SentenceTransformer(model_name)
        
        # Load data and embeddings
        self.load_data()
        self.load_or_create_embeddings()
    
    def load_data(self):
        """Load entitlements data from CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            required_columns = ['Entitlements', 'Description', 'Tags']
            
            if not all(col in self.df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Clean data
            self.df = self.df.dropna(subset=required_columns)
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
                    
                # Verify cache is still valid
                if (len(cache_data['embeddings']) == len(self.df) and 
                    cache_data['csv_modified'] == os.path.getmtime(self.csv_path)):
                    self.embeddings = cache_data['embeddings']
                    print("Loaded cached embeddings")
                    return
            except Exception as e:
                print(f"Error loading cached embeddings: {e}")
        
        # Create new embeddings
        print("Creating embeddings... This may take a moment.")
        self.embeddings = self.model.encode(self.df['combined_text'].tolist())
        
        # Cache embeddings
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
        """
        Search for relevant entitlements based on query
        
        Args:
            query: User query string
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of dictionaries with entitlement info and scores
        """
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top results above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            score = similarities[idx]
            if score >= similarity_threshold:
                results.append({
                    'entitlement': self.df.iloc[idx]['Entitlements'],
                    'description': self.df.iloc[idx]['Description'],
                    'tags': self.df.iloc[idx]['Tags'],
                    'similarity_score': float(score),
                    'relevance': self._get_relevance_label(score)
                })
        
        return results
    
    def _get_relevance_label(self, score):
        """Convert similarity score to relevance label"""
        if score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def format_results(self, results, max_results=None):
        """Format search results for display"""
        if not results:
            return "No relevant entitlements found. Try rephrasing your query or being more specific."
        
        if max_results:
            results = results[:max_results]
        
        output = f"\n{'='*60}\n"
        output += f"Found {len(results)} relevant entitlement(s):\n"
        output += f"{'='*60}\n"
        
        for i, result in enumerate(results, 1):
            output += f"\n{i}. {result['entitlement']}\n"
            output += f"   Relevance: {result['relevance']} ({result['similarity_score']:.3f})\n"
            output += f"   Description: {result['description']}\n"
            output += f"   Tags: {result['tags']}\n"
            output += f"   {'-'*50}\n"
        
        return output

class LlamaIntegration:
    """Integration with local Llama model for enhanced responses"""
    
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
        """Use Llama to provide contextual explanation"""
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

def create_sample_csv():
    """Create a sample CSV file for testing"""
    sample_data = {
        'Entitlements': [
            'DB_READ_PROD',
            'AWS_S3_BUCKET_ACCESS',
            'JENKINS_BUILD_DEPLOY',
            'GITLAB_REPO_ADMIN',
            'KUBERNETES_CLUSTER_ACCESS',
            'MONITORING_DASHBOARD_VIEW',
            'LOG_ANALYTICS_READ',
            'SECRETS_MANAGER_READ',
            'VPN_ACCESS_DEV',
            'JIRA_ADMIN_ACCESS'
        ],
        'Description': [
            'Read access to production databases for data analysis and reporting',
            'Access to AWS S3 buckets for file storage and retrieval operations',
            'Permission to trigger builds and deployments through Jenkins CI/CD',
            'Administrative access to GitLab repositories for code management',
            'Access to Kubernetes clusters for container orchestration and management',
            'View access to monitoring dashboards for system health tracking',
            'Read access to centralized logging system for troubleshooting',
            'Read access to secrets management system for configuration retrieval',
            'VPN access to development environment for remote work',
            'Administrative access to JIRA for project management and issue tracking'
        ],
        'Tags': [
            'database, production, read, analytics, reporting, data, SQL',
            'AWS, S3, storage, files, cloud, bucket, upload, download',
            'Jenkins, CI/CD, build, deploy, automation, pipeline, DevOps',
            'GitLab, git, repository, admin, code, version control, source',
            'Kubernetes, k8s, container, orchestration, pods, deployment',
            'monitoring, dashboard, metrics, alerts, system health, observability',
            'logs, logging, troubleshooting, debugging, analysis, centralized',
            'secrets, configuration, credentials, security, vault, passwords',
            'VPN, development, remote, network, access, dev environment',
            'JIRA, project management, issues, tickets, agile, scrum, admin'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('entitlements_test.csv', index=False)
    print("Created sample_entitlements.csv with sample data")

def main():
    parser = argparse.ArgumentParser(description='Entitlement Recommendation System')
    parser.add_argument('--csv', default='/Users/arnavchouhan/Documents/aimsbot/AIMS-bot/backend/entitlements.csv', 
                       help='Path to CSV file with entitlements')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample CSV file for testing')
    parser.add_argument('--max-results', type=int, default=10,
                       help='Maximum number of results to show')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Similarity threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_csv()
        return
    
    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        print("Use --create-sample to create a sample CSV file")
        return
    
    try:
        # Initialize recommender
        print("Initializing Entitlement Recommendation System...")
        recommender = EntitlementRecommender(args.csv)
        
        # Initialize Llama integration
        llama = LlamaIntegration()
        if llama.available:
            print("✓ Llama3 integration available")
        else:
            print("⚠ Llama3 not available (install Ollama and run 'ollama run llama3' for enhanced responses)")
        
        print("\n" + "="*60)
        print("ENTITLEMENT RECOMMENDATION CHATBOT")
        print("="*60)
        print("Describe your department, role, project, or specific needs.")
        print("Type 'quit' or 'exit' to end the session.")
        print("Type 'help' for usage examples.")
        print("="*60)
        
        while True:
            try:
                query = input("\n🤖 What entitlements do you need help with? ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("""
Example queries:
- "I'm a data analyst who needs to create reports from production data"
- "DevOps engineer setting up CI/CD pipeline for microservices"
- "Frontend developer working on React app deployment"
- "Database administrator managing user permissions"
- "Security team member auditing system access"
- "Project manager tracking development progress"
                    """)
                    continue
                
                if not query:
                    continue
                
                # Search for entitlements
                print("\n🔍 Searching for relevant entitlements...")
                results = recommender.search_entitlements(
                    query, 
                    top_k=args.max_results,
                    similarity_threshold=args.threshold
                )
                
                # Display results
                print(recommender.format_results(results, args.max_results))
                
                # Get Llama enhancement if available
                if results and llama.available:
                    print("\n💡 AI Insight:")
                    enhancement = llama.enhance_recommendations(query, results)
                    if enhancement:
                        print(enhancement)
                    print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
    
    except Exception as e:
        print(f"Failed to initialize system: {e}")

if __name__ == "__main__":
    main()