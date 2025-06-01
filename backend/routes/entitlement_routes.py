from app import app
from flask import request, jsonify
from models.entitlement_recommender import EntitlementRecommender

er = EntitlementRecommender(
    csv_path='/Users/arnavchouhan/Documents/aimsbot/AIMS-bot/backend/entitlements.csv',
    model_name='all-MiniLM-L6-v2',
    embeddings_cache='embeddings.pkl'
)

app.route('/recommend', methods=['POST'])
def recommend_entitlements():
    data = request.get_json()

    results = er.search_entitlements(
        query=data.get('query', '')
    )

    return jsonify({"results": results}), 200

