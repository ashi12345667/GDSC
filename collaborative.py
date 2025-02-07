import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
def load_data():
    file_paths = {
        'u_data': 'https://github.com/ashi12345667/GDSC/blob/main/u.data',
        'u_item': 'https://github.com/ashi12345667/GDSC/blob/main/u.item',
    }

    # Load ratings data
    df = pd.read_csv(file_paths['u_data'], sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df = df.drop(columns=['timestamp'])

    # Load movie info
    item_df = pd.read_csv(file_paths['u_item'], sep='|', encoding='latin-1', header=None,
                          names=['movie_id', 'movie_title'], usecols=[0, 1])

    return df, item_df

# Create user-item matrix
def create_user_item_matrix(df):
    user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
    return user_item_matrix

# Compute user similarity
def compute_similarity(user_item_matrix):
    user_similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Get recommendations
def get_user_based_recommendations(user_id, user_item_matrix, user_similarity, movie_dict, n_recommendations=5):
    if user_id not in user_similarity.index:
        return ["No recommendations available"]

    similar_users = user_similarity[user_id].sort_values(ascending=False).index[1:]
    watched_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] != 0].index

    recommended_movies = []
    for sim_user in similar_users:
        sim_user_ratings = user_item_matrix.loc[sim_user]
        highly_rated_movies = sim_user_ratings[sim_user_ratings >= 4]
        recommendations = highly_rated_movies[~highly_rated_movies.index.isin(watched_movies)]
        recommended_movies.extend(recommendations.index.tolist())
        if len(recommended_movies) >= n_recommendations:
            break

    return [movie_dict[movie_id] for movie_id in recommended_movies[:n_recommendations]]

import streamlit as st
from model import load_data, create_user_item_matrix, compute_similarity, get_user_based_recommendations

# Load Data
df, item_df = load_data()
movie_dict = dict(zip(item_df['movie_id'], item_df['movie_title']))
user_item_matrix = create_user_item_matrix(df)
user_similarity = compute_similarity(user_item_matrix)

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("User-Based Collaborative Filtering on MovieLens 100K Dataset")

# User Input
user_id = st.number_input("Enter User ID (1-943):", min_value=1, max_value=943, step=1)

if st.button("Get Recommendations"):
    recommendations = get_user_based_recommendations(user_id, user_item_matrix, user_similarity, movie_dict, 5)
    st.subheader("Recommended Movies:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")
