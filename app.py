from flask import Flask, request, jsonify
import joblib
import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import hstack, csr_matrix
import os

# Load the model and vectorizer
# Ensure the paths are correct relative to where your app is running.
model_path = 'relevance_score_xgboost_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
word2vec_model_path = 'word2vec_model_file'

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    word2vec_model = Word2Vec.load(word2vec_model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {str(e)}")

# Function to generate average word vectors
def get_average_word2vec(review, model, vector_size):
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the review from the request
        data = request.get_json(force=True)
        review = data.get('review')
        
        if not review:
            return jsonify({'error': 'Review text is missing'}), 400
        
        # Transform the review using the TF-IDF vectorizer
        X_tfidf = vectorizer.transform([review])
        
        # Generate Word2Vec features for the review
        X_word2vec = np.array([get_average_word2vec(review, word2vec_model, 100)])
        
        # Convert Word2Vec features to a sparse matrix
        X_word2vec_sparse = csr_matrix(X_word2vec)
        
        # Combine features
        X_combined = hstack([X_tfidf, X_word2vec_sparse])
        
        # Make prediction
        relevance_score = model.predict(X_combined)
        
        # Return the prediction as a JSON response
        return jsonify({'relevance_score': float(relevance_score[0])})
    
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
