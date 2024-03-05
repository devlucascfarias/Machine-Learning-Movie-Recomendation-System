import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('content/movies.csv')

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorize = TfidfVectorizer()
feature_vectors = vectorize.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

def recommend_movies(movie_name):

    list_of_all_titles = movies_data['title'].tolist()

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        return ["The movie you are searching for does not exist in our database. Please check the spelling or try with some other movies."]
    
    close_match = find_close_match[0]

    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    similarity_score = similarity[index_of_the_movie]

    sorted_similar_movies = sorted(list(enumerate(similarity_score)), key = lambda x:x[1], reverse = True)

    recommended_movies = []
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index==index]['title'].values[0]
        if (i<31):
            recommended_movies.append(title_from_index)
            i+=1
    return recommended_movies

st.set_page_config(page_title = "MRS - Movie Recommendation System",
                   layout = "centered",
                   page_icon = "🎥"
                   )

st.title('🎥 Movie Recommendation System')
movie_name = st.text_input('The system use AI to recommend movies based on your favourite movie.', placeholder = 'Enter the name of your favourite movie')
if st.button('Search'):
    recommended_movies = recommend_movies(movie_name)
    if recommended_movies == ["The movie you are searching for does not exist in our database. Please check the spelling or try with some other movies."]:
        st.write(recommended_movies[0])
    else:
        st.write(f"Top 30 recommended movies for {movie_name} are:")
        for i, movie in enumerate(recommended_movies, 1):
            st.write(f"{i}. {movie}")