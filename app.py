from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create the Flask application
app = Flask(__name__)

# Load the movie dataset
movies = pd.read_csv("movies.csv")

# Combine the movie title and genre
movies["features"] = movies["title"] + " " + movies["genres"]

# Convert text into numbers
cv = CountVectorizer(stop_words="english")
count_matrix = cv.fit_transform(movies["features"])

# Calculate similarity between movies
similarity = cosine_similarity(count_matrix)


def recommend(movie_name):
    movie_name = movie_name.lower()

    for index, title in enumerate(movies["title"]):
        if movie_name == title.lower():

            scores = list(enumerate(similarity[index]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            recommendations = []

            for movie in scores[1:6]:
                recommendations.append(movies.iloc[movie[0]].title)

            return recommendations

    return ["Movie not found. Try another title."]


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":
        movie = request.form["movie"]
        recommendations = recommend(movie)

    return render_template(
        "index.html",
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)