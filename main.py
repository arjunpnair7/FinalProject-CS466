import pandas as pd
from Models.knn import predict_labels
from Models.global_alignment import global_alignment
from Plots.Utils import GAP_PENALTY, SIMILARITY_THRESHOLD, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY, FILE_PATH, MovieEntry



df = pd.read_csv(FILE_PATH)
movie_entries = [MovieEntry(row['Overview'], row['Genre']) for _, row in df.iterrows()]

training_set = []
training_set_labels = []

for i in range(900):
    training_set.append(movie_entries[i])
    training_set_labels.append(movie_entries[i].genre)

test_set = []
test_set_labels = []
for i in range(900, 925):
    test_set.append(movie_entries[i])
    test_set_labels.append(movie_entries[i].genre)

# Set up an evaluation function for KNN
def check_genre_match(predicted_label, true_label):
        for i in range(len(predicted_label)):
             if predicted_label[i] in true_label:
                  return 1
        return 0


rewards = (GAP_PENALTY, SIMILARITY_THRESHOLD, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY)

# Uncomment below to test accuracy with various K values
K_val_accuracies = []
K_values = [10, 20, 30, 40, 50, 60, 70]
for k_val in K_values:
     pass
     
     _, accuracy = predict_labels(training_set, training_set_labels, test_set, test_set_labels, global_alignment, check_genre_match, k_val, rewards)
     K_val_accuracies.append(accuracy)

print("K Values: ")
print(K_values, " : ", K_val_accuracies)
# for i in range(len(K_values)):
#      print(K_values[i], " : ", K_val_accuracies[i])


# Uncomment below to test accuracy with various threshold values
# threshold_accuracies = []
# threshold_values = [.2, .4, .6, .8, 1]
# for threshold_value in threshold_values:
#     #  print("K_VAL, LINE 57: ", k_val)
#      rewards = (GAP_PENALTY, threshold_value, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY)
#      _, accuracy = predict_labels(training_set, training_set_labels, test_set, test_set_labels, global_alignment, check_genre_match, 30, rewards)
#      threshold_accuracies.append(accuracy)
# print("Threshold Values: ")
# print(threshold_values, " : ", threshold_accuracies)


# Uncomment below to test accuracy with various mistmatch rewards
# mismatch_accuracies = []
# mismatch_values = [-.25, -.5, -.75, -1, -1.25, -1.5, -1.75]
# for mismatch_value in mismatch_values:
#      rewards = (GAP_PENALTY, SIMILARITY_THRESHOLD, SIMILARITY_REWARD, mismatch_value)
#      _, accuracy = predict_labels(training_set, training_set_labels, test_set, test_set_labels, global_alignment, check_genre_match, 30, rewards)
#      mismatch_accuracies.append(accuracy)
# print("Mismatch Values: ")
# print(mismatch_values, " : ", mismatch_accuracies)
     
