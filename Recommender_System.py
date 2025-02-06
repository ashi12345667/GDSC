pip install implicit
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

movies= pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits= pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

movies=movies.merge(credits, on ='title')
movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()

import ast
def convert_string(object):
    L=[]
    for i in ast.literal_eval(object):
        L.append(i['name'])
    return L
movies['genres']=movies['genres'].apply(convert_string)
movies['keywords']=movies['keywords'].apply(convert_string)

import ast
def convert_cast(object):
    L=[]
    counter=0
    for i in ast.literal_eval(object):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break 
    return L

movies['cast']=movies['cast'].apply(convert_cast)


def convert_crew(object):
    L=[]
    for i in ast.literal_eval(object):
        if i['job']=='Director':
            L.append(i['name']) 
            break
    return L
movies['crew']=movies['crew'].apply(convert_crew)
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies['overview']=movies['overview'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags']=movies['overview']+movies['keywords']+movies['cast']+movies['crew']+movies['genres']
new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words = 'english')
vectors = cv.fit_transform(new_df['tags']).toarray()

vectors[0]
vectors.shape

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommand(movies):
    movies_index = new_df[new_df['title'] == movies].index[0]
    distances = similarity[movies_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True, key=lambda x:x[1])[1:6]

    for i in movies_list:
            print(new_df.iloc[i[0]].title)

# ---------------- IMPLICIT COLLABORATIVE FILTERING ----------------

# Simulate user interactions (implicit feedback)
user_ids = np.random.randint(1, 100, size=len(movies))  # Assign random users
interaction_matrix = pd.DataFrame({'user_id': user_ids, 'movie_id': movies['movie_id']})

# Create sparse interaction matrix
sparse_matrix = csr_matrix((np.ones(len(interaction_matrix)), 
                            (interaction_matrix['user_id'], interaction_matrix['movie_id'])))

# Train ALS model
model = AlternatingLeastSquares(factors=50, regularization=0.1)
model.fit(sparse_matrix.T)

# Function to recommend movies using Implicit Collaborative Filtering
def recommend_cf(user_id, num_recommendations=5):
    if user_id not in interaction_matrix['user_id'].values:
        return ["User not found. Try another ID."]
    
    user_movies = interaction_matrix[interaction_matrix['user_id'] == user_id]['movie_id'].tolist()
    recommendations = model.recommend(user_id, sparse_matrix, N=num_recommendations)
    
    return movies[movies['movie_id'].isin([rec[0] for rec in recommendations])]['title'].values


# ---------------- HYBRID RECOMMENDATION ----------------

def hybrid_recommendation(user_id, movie_title, num_recommendations=5):
    content_recs = recommend_content(movie_title)
    cf_recs = recommend_cf(user_id, num_recommendations)
    
    hybrid_recs = list(set(content_recs + list(cf_recs)))[:num_recommendations]
    return hybrid_recs



  
