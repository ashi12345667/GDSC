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

# Load dataset
file_url1 = "https://raw.githubusercontent.com/ashi12345667/GDSC/main/tmdb_5000_movies.csv"
movies = pd.read_csv(file_url1)

file_url2 = "https://raw.githubusercontent.com/ashi12345667/GDSC/main/compressed_data.csv.gz"
credits = pd.read_csv(file_url2)

# movies = pd.read_csv('https://github.com/ashi12345667/GDSC/blob/main/tmdb_5000_movies.csv')
# credits = pd.read_csv('https://github.com/ashi12345667/GDSC/blob/main/compressed_data.csv.gz')

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

# ---------------- IMPLICIT COLLABORATIVE FILTERING ----------------

# Simulate user interactions (implicit feedback)
user_ids = np.random.randint(1, 100, size=len(movies))
interaction_matrix = pd.DataFrame({'user_id': user_ids, 'movie_id': movies['movie_id']})

# Create sparse interaction matrix
sparse_matrix = csr_matrix((np.ones(len(interaction_matrix)), 
                            (interaction_matrix['user_id'], interaction_matrix['movie_id'])))

# Train ALS model
model = AlternatingLeastSquares(factors=50, regularization=0.1)
model.fit(sparse_matrix.T)

# Collaborative Filtering Recommendation
def recommend_cf(user_id, num_recommendations=5):
    if user_id not in interaction_matrix['user_id'].values:
        return ["User not found. Try another ID."]
    
    recommendations = model.recommend(user_id, sparse_matrix, N=num_recommendations)
    return movies[movies['movie_id'].isin([rec[0] for rec in recommendations])]['title'].values

# ---------------- HYBRID RECOMMENDATION ----------------

def hybrid_recommendation(user_id, movie_title, num_recommendations=5):
    content_recs = recommend_content(movie_title)
    cf_recs = recommend_cf(user_id, num_recommendations)
    
    hybrid_recs = list(set(content_recs + list(cf_recs)))[:num_recommendations]
    return hybrid_recs
