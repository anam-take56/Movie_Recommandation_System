#importing the libreries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

#loading the dataset
movies=pd.read_csv(r"E:\Rouhi_DEVELOPMENT\3)Python+Pyspark\mid\pythonProject\movies.csv")
ratings=pd.read_csv(r"E:\Rouhi_DEVELOPMENT\3)Python+Pyspark\mid\pythonProject\ratings.csv")


def popularity_recommender(genre, ratings_threshold, num_recommendations):
    # Filter movies by genre
    genre_movies = movies[movies['genres'].str.contains(genre, case=False)]

    # Filter movies by ratings threshold
    high_rated_movies = ratings.groupby('movieId')['rating'].agg(['count', 'mean']).reset_index()
    high_rated_movies = high_rated_movies[high_rated_movies['count'] >= ratings_threshold]

    # Merge datasets
    genre_high_rated_movies = pd.merge(genre_movies, high_rated_movies, on='movieId')

    # Sort by popularity (average rating)
    sorted_movies = genre_high_rated_movies.sort_values(by='mean', ascending=False)

    # Recommend top N movies
    top_recommendations = sorted_movies.head(num_recommendations)

    return top_recommendations[['title', 'mean', 'count']]


def content_recommender(movie_title, num_recommendations):
    # Find the index of the movie with the given title
    movie_index = movies[movies['title'] == movie_title].index[0]

    # Create a TF-IDF Vectorizer for movie genres
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'].fillna(''))

    # Calculate the cosine similarity between movies
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Get the similarity scores for the given movie
    similarity_scores = list(enumerate(cosine_similarities[movie_index]))

    # Sort movies based on similarity scores
    similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]

    # Get the top 'num_recommendations' similar movies
    top_movies_indices = [index for index, _ in similar_movies[:num_recommendations]]

    # Display the final result
    recommendations = movies.iloc[top_movies_indices][['title']].reset_index(drop=True)

    return recommendations


def collaborative_recommender(target_user_id, num_similar_users, num_recommendations):
    # Create a user-item matrix
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix)

    # Identify K similar users for the target user
    similar_users_indices = user_similarity[target_user_id - 1].argsort()[::-1][1:num_similar_users+1]

    # Predict ratings for unrated movies for the target user
    target_user_ratings = user_item_matrix.loc[target_user_id]
    predicted_ratings = user_item_matrix.iloc[similar_users_indices].mean(axis=0)

    # Filter unrated movies
    unrated_movies = target_user_ratings[target_user_ratings == 0].index

    # Sort and recommend top N movies
    recommendations = predicted_ratings[unrated_movies].sort_values(ascending=False).head(num_recommendations)

    # Create a DataFrame with movieId and predicted rating
    recommendations_df = pd.DataFrame({'movieId': recommendations.index, 'predicted_rating': recommendations.values})

    # Merge with movies DataFrame to get movie titles
    recommendations_df = pd.merge(recommendations_df, movies[['movieId', 'title']], on='movieId')

    return recommendations_df[['title', 'predicted_rating']]
