import pandas as pd
import numpy as np
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.neighbors import NearestNeighbors


# Load dataset
file_url1 = "https://raw.githubusercontent.com/ashi12345667/GDSC/main/tmdb_5000_movies.csv"
movies = pd.read_csv(file_url1)


file_url2 = "https://raw.githubusercontent.com/ashi12345667/GDSC/main/compressed_data.csv.gz"
credits = pd.read_csv(file_url2)


# Merge datasets
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)


# Convert genres, keywords, cast, crew from JSON-like format to lists
def convert_string(object):
    L = []
    for i in ast.literal_eval(object):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert_string)
movies['keywords'] = movies['keywords'].apply(convert_string)


# Extract top 3 cast members
def convert_cast(object):
    L = []
    for i in ast.literal_eval(object)[:3]:  # Take first 3 actors
        L.append(i['name'])
    return L

movies['cast'] = movies['cast'].apply(convert_cast)


# Extract director name
def convert_crew(object):
    for i in ast.literal_eval(object):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['crew'] = movies['crew'].apply(convert_crew)


# Process overview, genres, cast, crew, and keywords
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# Create a new feature 'tags' combining important text-based features
movies['tags'] = movies['overview'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['genres']


# Prepare final DataFrame
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# Stemming for better text matching
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])
new_df['tags'] = new_df['tags'].apply(stem)


# Convert tags to numerical vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vectors)

# Content-Based Recommendation
def recommend_content(movie_title):
    if movie_title not in new_df['title'].values:
        return ["Movie not found. Try another title."]
    
    movie_index = new_df[new_df['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    return [new_df.iloc[i[0]].title for i in movies_list]

# IMPLICIT COLLABORATIVE FILTERING
# Load Data
def load_data():
    file_paths = {
    'u_data': 'https://raw.githubusercontent.com/ashi12345667/GDSC/main/u.data',
    'u_item': 'https://raw.githubusercontent.com/ashi12345667/GDSC/main/u.item',
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
    return pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


def compute_similarity(user_item_matrix):
    if user_item_matrix.shape[0] == 0 or user_item_matrix.shape[1] == 0:
        print("Error: Empty user-item matrix. Cannot compute similarity.")
        return pd.DataFrame()  # Return an empty DataFrame to avoid crashing
    
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

# Load Data
df, item_df = load_data()
movie_dict = dict(zip(item_df['movie_id'], item_df['movie_title']))
user_item_matrix = create_user_item_matrix(df)
user_similarity = compute_similarity(user_item_matrix)


# Streamlit UI for interactive user input
import streamlit as st
st.title("🎬 Movie Recommender System")

st.sidebar.header("Choose Recommendation Type")
rec_type = st.sidebar.selectbox("Select Type", ["Content-Based", "Collaborative (KNN)"])

if rec_type == "Content-Based":
    movie_title = st.selectbox("Select a Movie", movies['title'].values)
    if st.button("Recommend"):
        recommendations1 = recommend_content(movie_title)
        st.write("### Recommended Movies:")
        for movie in recommendations1:
            st.write(f"✅ {movie}")


elif rec_type == "Collaborative (KNN)":
    st.write("User-Based Collaborative Filtering on MovieLens 100K Dataset")
    user_id = st.number_input("Enter User ID (1-943):", min_value=1, max_value=943, step=1)
    
    if st.button("Get Recommendations"):
        recommendations = get_user_based_recommendations(user_id, user_item_matrix, user_similarity, movie_dict, 5)
        st.subheader("Recommended Movies:")
        for i, movie in enumerate(recommendations, 1):
             st.write(f"✅ {movie}")
       
