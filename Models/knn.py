# from global_alignment import global_alignment
from collections import Counter

def predict_labels(training_set, training_labels, test_set, test_labels, similarity_function, evaluation_function, K, rewards):

    # print("TRAINING SET: ", training_set)
    # print("TEST SET: ", test_set)

    def predict_single_label(test_point):
            scores = []
            for index, cur_entry in enumerate(training_set):
                similarity_score = similarity_function(test_point, cur_entry, rewards)
                scores.append((index, similarity_score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [score[0] for score in scores[:K]]
            top_k_labels = []
            # print("SCORES: ", scores)
            for index in top_k_indices:
                # print(training_labels[index])
                top_k_labels.append(tuple(training_labels[index]))

            # print("TOP K LABELS: ", top_k_labels)
            # print(top_k_labels)

            label_counts = Counter(tuple(top_k_labels))
            predicted_label = label_counts.most_common(1)[0][0]
            # print("LINE 20: ", label_counts)
            # print("LINE 21: ", predicted_label)

            return predicted_label
    

    correct_count = 0
    total = len(test_set)
    predicted_label_list = []
    # print(len(test_set), print(len(training)))
    for idx in range(len(test_set)):
        predicted_label = predict_single_label(test_set[idx])
        # print(predicted_label)
        predicted_label_list.append(predicted_label)

        true_label = test_labels[idx]
        correct_count += evaluation_function(predicted_label, true_label)

    return predicted_label_list, (correct_count / total)
    
    

# class KNN:

#     def __init__(self, movie_entries, test_movie_entries, k=50):
#         self.movie_entries = movie_entries
#         self.k = k
#         self.test_movie_entries = test_movie_entries


#     def evaluate_performance(self):
#         points = 0
#         for test_entry in self.test_movie_entries:
#             predicted_genres = self.predict_movie_genre(test_entry)
#             if self.check_match(predicted_genres, test_entry.genre):
#                 points += 1
        
#         print("CORRECT: ", points)
#         print("INCORRECT: ",len(self.test_movie_entries) - points)
#         print("ACCURACY: ", points / (len(self.test_movie_entries)))


#     def predict_movie_genre(self, new_movie_entry):
#         scores = []
#         idx = 0
#         for cur_entry in self.movie_entries:
#             idx += 1
#             print("PROCESS ENTRY: ", idx, "...")
#             similarity_score = global_alignment(new_movie_entry.overview, cur_entry.overview)
#             scores.append((cur_entry, similarity_score))

#         scores.sort(key=lambda x: x[1], reverse=True)
#         top_k_movies = [movie[0] for movie in scores[:self.k]]
#         all_genres = []
#         for movie in top_k_movies:
#             all_genres.extend(movie.genre)

#         genre_counts = Counter(all_genres)

#         top_3_genres = genre_counts.most_common(3)
#         top_3_genres_list = [genre for genre, _ in top_3_genres]
#         # print("Top 3 Most Frequent Genres:", top_3_genres_list)

#         return top_3_genres_list


#     def check_match(self, predictions, true_labels):
#         matches = 0
#         for label in true_labels:
#             if label in predictions:
#                 matches += 1
#         if matches/len(true_labels) >= .5:
#             return True
#         else:
#             return False







