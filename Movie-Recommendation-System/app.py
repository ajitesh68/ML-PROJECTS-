from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load preprocessed data
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
movies = pd.read_csv("data/ml-latest-small/movies.csv")
merged_data = pd.merge(ratings, movies, on="movieId")

# Create two matrices: one raw (with NaNs), one filled (for similarity calc)
user_movie_matrix_raw = merged_data.pivot_table(index="userId", columns="title", values="rating")
user_movie_matrix = user_movie_matrix_raw.fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_similarity_df.index:
        return {"error": "User ID not found"}

    # Find similar users (exclude self)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    
    # Get ratings from similar users
    similar_users_ratings = user_movie_matrix.loc[similar_users]
    
    # Calculate weighted ratings
    weighted_ratings = similar_users_ratings.T.dot(user_similarity_df[user_id].loc[similar_users])
    weighted_ratings = weighted_ratings / user_similarity_df[user_id].loc[similar_users].sum()

    # Use the raw matrix (with NaNs) to find movies the user hasn't rated
    user_ratings_raw = user_movie_matrix_raw.loc[user_id]
    unrated_movies = user_ratings_raw[user_ratings_raw.isna()].index
    recommendations = weighted_ratings[unrated_movies].sort_values(ascending=False).head(num_recommendations)

    return recommendations.index.tolist()

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id_str = request.args.get("userId")
    if user_id_str is None:
        return jsonify({"error": "Missing userId"}), 400
    
    try:
        user_id = int(user_id_str)
        num_recommendations = int(request.args.get("num", 5))
        recommendations = recommend_movies(user_id, num_recommendations)
        return jsonify({"userId": user_id, "recommendations": recommendations})
    except ValueError:
        return jsonify({"error": "Invalid userId or num"}), 400

@app.route('/')
def home():
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run(debug=True)
