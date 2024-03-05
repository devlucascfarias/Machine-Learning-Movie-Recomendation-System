# Machine-Learning-Movie-Recomendation-System

This is a movie recommendation system built with Streamlit, a Python library for creating web applications, and Scikit-learn, a machine learning library. The system uses a dataset of movies and their associated metadata, such as genres, keywords, tagline, cast, and director.

Here's a step-by-step description of the code:

Import necessary libraries: Streamlit for the web app, pandas for data manipulation, numpy for numerical operations, difflib for finding close matches of movie names, and Scikit-learn for feature extraction and similarity calculation.

Load the movies dataset from a CSV file.

Select the features to be used for the recommendation system: 'genres', 'keywords', 'tagline', 'cast', 'director'. Fill any missing values in these features with an empty string.

Combine the selected features into a single string for each movie.

Use the TfidfVectorizer from Scikit-learn to convert the combined features into a matrix of TF-IDF features.

Calculate the cosine similarity between each pair of movies based on their TF-IDF features.

Define a function recommend_movies that takes a movie name as input and returns a list of recommended movies. This function first finds the closest match of the input movie name in the dataset. If no match is found, it returns an error message. Otherwise, it finds the 30 movies that have the highest cosine similarity with the input movie.

Set up the Streamlit web app with a title, a text input for the user to enter their favorite movie, and a button to start the recommendation. When the button is clicked, the app calls the recommend_movies function with the user's input and displays the recommended movies or the error message.

This system uses a content-based recommendation approach, where recommendations are based on the similarity between the content (in this case, the metadata) of different items (movies).
