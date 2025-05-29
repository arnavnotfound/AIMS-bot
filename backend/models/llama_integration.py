import requests
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
