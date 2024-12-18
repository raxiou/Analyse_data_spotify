from knn.knn import knn
from knn.knn import prepaData
import knn.knnTest as knnTest
import pandas as pd
import sys

def __main__():
    test_knn = input("Do you want to test the KNN model? (yes/no): ").strip().lower()
    if test_knn == 'yes':
        knnTest.__main__()
    else:
        print("KNN model will not be tested.")

    # Ask the user if they want to use the KNN model
    use_knn = input("Do you want to use the KNN model? (yes/no): ").strip().lower()

    if use_knn == 'yes':
        # Load the data
        data = pd.read_csv("data/spotify_songs_with_popularity.csv", low_memory=False)

        # Prepare the data
        data, track_names, Popularity_Category, q = prepaData(data)

        # Define a query
        query = data.iloc[0]

        # Predict the popularity category of the query
        k = 10
        predict = knn(data, query, k, Popularity_Category)
        print(f"Predicted popularity category: {predict}")
    else:
        print("KNN model will not be used.")
    

    use3song = input("Do you want test the model on 3 songs? (yes/no): ").strip().lower()

    if use3song == 'yes':
        list_song_id = ['711MglQhnhOF3UAiAs9A59', '379GgTu7WeNT4Xoak52p3E', '3uouaAVXpQR3X8RYkJyitQ']
        
        dataBase = pd.read_csv("data/spotify_songs_with_popularity.csv", low_memory=False)

        for song in list_song_id:
            data = dataBase
            data, track_names, Popularity_Category, query = prepaData(data, song)
            k = 10
            predict = knn(data, query, k, Popularity_Category)
            print(f"Predicted popularity category for {song}: {predict}")

if __name__ == "__main__":
    __main__()