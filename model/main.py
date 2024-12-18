from knn.knn import knn
from knn.knn import prepaData
import pandas as pd

def __main__():
    # Load the data
    data = pd.read_csv("data/spotify_songs_with_popularity.csv", low_memory=False)

    # Prepare the data
    data, track_names, Popularity_Category = prepaData(data)

    # Define a query
    query = data.iloc[0]

    # Predict the popularity category of the query
    k = 5
    predict = knn(data, query, k, Popularity_Category)
    print(f"Predicted popularity category: {predict}")

if __name__ == "__main__":
    __main__()