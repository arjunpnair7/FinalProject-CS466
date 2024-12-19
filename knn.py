from global_alignment import global_alignment
from collections import Counter

class KNN:

    def __init__(self, movie_entries, test_movie_entries, k=50):
        self.movie_entries = movie_entries
        self.k = k
        self.test_movie_entries = test_movie_entries


    def evaluate_performance(self):
        points = 0
        for test_entry in self.test_movie_entries:
            predicted_genres = self.predict_movie_genre(test_entry)
            if self.check_match(predicted_genres, test_entry.genre):
                points += 1
        
        print("CORRECT: ", points)
        print("INCORRECT: ",len(self.test_movie_entries) - points)
        print("ACCURACY: ", points / (len(self.test_movie_entries)))


    def predict_movie_genre(self, new_movie_entry):
        scores = []
        idx = 0
        for cur_entry in self.movie_entries:
            idx += 1
            print("PROCESS ENTRY: ", idx, "...")
            similarity_score = global_alignment(new_movie_entry.overview, cur_entry.overview)
            scores.append((cur_entry, similarity_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_movies = [movie[0] for movie in scores[:self.k]]
        all_genres = []
        for movie in top_k_movies:
            all_genres.extend(movie.genre)

        genre_counts = Counter(all_genres)

        top_3_genres = genre_counts.most_common(3)
        top_3_genres_list = [genre for genre, _ in top_3_genres]
        # print("Top 3 Most Frequent Genres:", top_3_genres_list)

        return top_3_genres_list


    def check_match(self, predictions, true_labels):
        matches = 0
        for label in true_labels:
            if label in predictions:
                matches += 1
        if matches/len(true_labels) >= .5:
            return True
        else:
            return False







