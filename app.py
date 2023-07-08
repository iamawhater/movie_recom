from flask import Flask, jsonify, request, render_template
import pandas as pd
from surprise import Reader, Dataset, KNNBasic

app = Flask(__name__)

movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv('ratings.csv')

merged_df = pd.merge(movies_df, ratings_df, on='movieId')
merged_df.drop(['timestamp', 'genres', 'title'], axis=1, inplace=True)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(merged_df[['userId', 'movieId', 'rating']], reader)

model = KNNBasic(sim_options={'user_based': True})
trainset = data.build_full_trainset()
model.fit(trainset)

def get_recommendations(userid, n_recommendations):
    movie_ids = movies_df['movieId'].unique()
    rated_movies = merged_df[merged_df['userId'] == userid]['movieId']
    unrated_movies = [mid for mid in movie_ids if mid not in rated_movies.values]

    predicted_ratings = []
    for mid in unrated_movies:
        pred = model.predict(userid, mid)
        movie_name = movies_df[movies_df['movieId'] == mid]['title'].values[0]
        predicted_ratings.append((movie_name, pred.est))

    predicted_ratings.sort(key=lambda x: x[1], reverse=True)  # Sort by rating in descending order
    return predicted_ratings[:n_recommendations]  # Return the top n recommendations

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendation():
    user_id = int(request.form.get('user_id'))
    n = int(request.form.get('n'))
    recommendations = get_recommendations(user_id, n)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run()
