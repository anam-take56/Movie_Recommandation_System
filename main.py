# Importing the libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from app import popularity_recommender, content_recommender, collaborative_recommender

# Loading the dataset
movies = pd.read_csv(r"E:\Rouhi_DEVELOPMENT\3)Python+Pyspark\mid\pythonProject\movies.csv")
ratings = pd.read_csv(r"E:\Rouhi_DEVELOPMENT\3)Python+Pyspark\mid\pythonProject\ratings.csv")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('main.html')


# Handling the form submission to select the recommendation type
@app.route('/select-recommendation', methods=['POST'])
def select_recommendation():
    recommendation_type = request.form.get('recommendationType')

    if recommendation_type == 'popularity':
        return render_template('Popularity.html')
    elif recommendation_type == 'content':
        return render_template('Content.html')
    elif recommendation_type == 'collaborative':
        return render_template('collaborative.html')


# Function for Popularity-Based Recommendations
@app.route('/popularity', methods=['POST'])
def popularity_recommendations():
    genre = request.form.get('genre')
    ratings_threshold = int(request.form.get('ratings_threshold'))
    num_recommendations = int(request.form.get('num_recommendations'))

    recommendations_df = popularity_recommender(genre, ratings_threshold, num_recommendations)
    # Check if recommendations_df is not empty before rendering the template
    if not recommendations_df.empty:
        recommendations_list = recommendations_df['title'].tolist()
        return render_template('result_popularity.html', recommendations=recommendations_list)
    else:
        return render_template('result_popularity.html', recommendations=None)


# Function for Content-Based Recommendations
@app.route('/content', methods=['POST'])
def content_recommendations():
    movie_title = request.form.get('movie_title')
    num_recommendations = int(request.form.get('num_recommendations'))
    recommendations_df = content_recommender(movie_title, num_recommendations)

    # Check if recommendations_df is not empty before rendering the template
    if not recommendations_df.empty:
        recommendations_list = recommendations_df['title'].tolist()
        return render_template('content_result.html', recommendations=recommendations_list)
    else:
        return render_template('content_result.html', recommendations=None)


# Function for Collaborative-Based Recommendations
@app.route('/collaborative', methods=['POST'])
def collaborative_recommendations():
    user_id = int(request.form.get('user_id'))
    num_similar_users = int(request.form.get('num_similar_users'))
    num_recommendations = int(request.form.get('num_recommendations'))
    recommendations_df = collaborative_recommender(user_id, num_similar_users, num_recommendations)

    # Check if recommendations_df is not empty before rendering the template
    if not recommendations_df.empty:
        recommendations_list = recommendations_df['title'].tolist()
        return render_template('collaborative_result.html', recommendations=recommendations_list)
    else:
        return render_template('collaborative_result.html', recommendations=None)



if __name__ == '__main__':
    app.run(debug=True)
