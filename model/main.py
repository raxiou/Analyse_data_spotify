from knn.knn import knn
from knn.knn import prepaData
import knn.knnTest as knnTest
import pandas as pd
import sys

def __main__():
    test_knn = input("Do you want to test the KNN model? (yes/no): ").strip().lower()
    if test_knn == 'yes':
        knnTest.__main__()
        sys.exit()
    else:
        print("KNN model will not be tested.")

    # Ask the user if they want to use the KNN model
    use_knn = input("Do you want to use the KNN model? (yes/no): ").strip().lower()

    if use_knn == 'yes':
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
    else:
        print("KNN model will not be used.")
        sys.exit()

if __name__ == "__main__":
    __main__()