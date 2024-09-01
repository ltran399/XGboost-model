import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from gensim.models import Word2Vec
import numpy as np
from scipy.stats import randint, uniform
import joblib
from scipy.sparse import hstack
# Step 1: Load the dataset
df = pd.read_csv('combined_reviews_with_refined_relevance_scores_v2.csv')

# Ensure that the 'Review' column exists
if 'Review' not in df.columns:
    raise ValueError("The DataFrame does not contain a 'Review' column.")

# Step 2: Handle NaN values in the 'Review' column
df['Review'] = df['Review'].fillna('')  # Replace NaN with empty string

# Step 3: TF-IDF Vectorization with N-Grams
vectorizer = TfidfVectorizer(max_features=2378, ngram_range=(1, 3))  # Ensure max_features matches across training and prediction
X_tfidf = vectorizer.fit_transform(df['Review'])

# Save the TF-IDF vectorizer for later use
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Step 4: Word Embeddings with Word2Vec
# Tokenize reviews into words
tokenized_reviews = [review.split() for review in df['Review']]

# Train a Word2Vec model on the tokenized reviews
word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save('word2vec_model_file')

# Create a function to generate the average word vectors for a review
def get_average_word2vec(review, model, vector_size):
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Apply the function to get Word2Vec features for each review
X_word2vec = np.array([get_average_word2vec(review, word2vec_model, 100) for review in df['Review']])

# Step 5: Combine TF-IDF features and Word2Vec features
X_combined = hstack([X_tfidf, X_word2vec])

# Step 6: Define target variable
y = df['Relevance_Score']

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Step 8: Initialize the XGBoost Regressor
model = XGBRegressor(random_state=42)

# Step 9: Define the hyperparameter distribution for RandomizedSearchCV
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 1.0),
    'colsample_bytree': uniform(0.6, 1.0),
}

# Step 10: Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,  # Number of parameter settings that are sampled
    scoring='neg_mean_squared_error',
    cv=3,  # 3-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Step 11: Fit the model with RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best parameters from RandomizedSearchCV
best_params = random_search.best_params_
print(f"Best parameters: {best_params}")

# Use the best model to predict
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Step 12: Save the best model and Evaluate it
joblib.dump(best_model, 'relevance_score_xgboost_model.pkl')

mse = mean_squared_error(y_test, y_pred)
print(f'Improved Mean Squared Error with N-Grams and Word Embeddings: {mse}')