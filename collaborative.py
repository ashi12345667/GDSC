import pandas as pd
import numpy as np
import pickle
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.neighbors import NearestNeighbors

def load_data():
    # Load ratings data
    ratings_path = "https://github.com/ashi12345667/GDSC/blob/main/u.data"
    columns = ["userId", "movieId", "rating", "timestamp"]
    df = pd.read_csv(ratings_path, sep="\t", names=columns)

    # Load movies data
    movies_path = "https://github.com/ashi12345667/GDSC/blob/main/u.item"
    movie_columns = ["movieId", "title"]
    movies_df = pd.read_csv(movies_path, sep="|", encoding="latin-1", usecols=[0, 1], names=movie_columns)

    # Convert movieId to integer in both dataframes
    df["movieId"] = df["movieId"].astype(int)
    movies_df["movieId"] = movies_df["movieId"].astype(int)

    # Merge the datasets
    df = df.merge(movies_df, on="movieId")

    return df, movies_df

# Load data
df, movies_df = load_data()


# Create a User-Item Interaction Matrix
pivot_table = df.pivot(index="userId", columns="title", values="rating").fillna(0)

# Compute Cosine Similarity
cosine_sim = cosine_similarity(pivot_table)

# KNN Model
knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
knn.fit(pivot_table)

def recommend_cf(user_id, n=10):
    user_index = user_id - 1  # Adjust for zero-indexing
    distances, indices = knn.kneighbors([pivot_table.iloc[user_index]], n_neighbors=n+1)
    
    # Get similar users
    similar_users = indices.flatten()[1:]
    
    # Find movies watched by similar users
    recommended_movies = []
    for sim_user in similar_users:
        top_movies = df[df["userId"] == sim_user]["title"].value_counts().index[:3]
        recommended_movies.extend(top_movies)
    
    return list(set(recommended_movies))[:n]

import streamlit as st
st.title("🎬 Movie Recommender System")

st.sidebar.header("Choose Recommendation Type")
rec_type = st.sidebar.selectbox("Select Type", ["Content-Based", "Collaborative (Implicit)", "Hybrid"])


if rec_type == "Collaborative (KNN)":
    user_id = st.number_input("Enter User ID", min_value=1, max_value=int(df['userId'].max()), step=1)
    if st.button("Recommend"):
        recommendations2 = recommend_cf(user_id)
        st.write("### Recommended Movies:")
        for movie in recommendations2:
            st.write(f"✅ {movie}")
