import os
import argparse
from models.entitlement_recommender import EntitlementRecommender
from models.llama_integration import LlamaIntegration

def main():
    parser = argparse.ArgumentParser(description='Entitlement Recommendation System')
    parser.add_argument('--csv', default='/Users/arnavchouhan/Documents/aimsbot/AIMS-bot/backend/entitlements.csv', help='Path to CSV file with entitlements')
    parser.add_argument('--create-sample', action='store_true',help='Create a sample CSV file for testing')
    parser.add_argument('--max-results', type=int, default=10,help='Maximum number of results to show')
    parser.add_argument('--threshold', type=float, default=0.3,help='Similarity threshold (0.0-1.0)')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        print("Use --create-sample to create a sample CSV file")
        return
    
    try:
        print("Initializing Entitlement Recommendation System...")
        recommender = EntitlementRecommender(args.csv)
        
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
        
        while True:
            try:
                query = input("\n🤖 What entitlements do you need help with? \n").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
            
                print("\n🔍 Searching for relevant entitlements...")
                results = recommender.search_entitlements(
                    query, 
                    top_k=args.max_results,
                    similarity_threshold=args.threshold
                )
                
                print(recommender.format_results(results, args.max_results))
                
                # if results and llama.available:
                #     print("\n💡 AI Insight:")
                #     enhancement = llama.enhance_recommendations(query, results)
                #     if enhancement:
                #         print(enhancement)
                #     print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
    
    except Exception as e:
        print(f"Failed to initialize system: {e}")
