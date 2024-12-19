import pandas as pd
from knn import predict_labels
from global_alignment import global_alignment
import Utils



df = pd.read_csv(Utils.FILE_PATH)
movie_entries = [Utils.MovieEntry(row['Overview'], row['Genre']) for _, row in df.iterrows()]


training_set = []
training_set_labels = []

for i in range(900):
    training_set.append(movie_entries[i])
    training_set_labels.append(movie_entries[i].genre)

test_set = []
test_set_labels = []
for i in range(900, 905):
    test_set.append(movie_entries[i])
    test_set_labels.append(movie_entries[i].genre)

# Set up an evaluation function for KNN
def check_genre_match(predicted_label, true_label):
        for i in range(len(predicted_label)):
             if predicted_label[i] in true_label:
                  return 1
        return 0

# predict_labels returns predicted labels but we do not need it
# Note: We are passing in global_alignment here as it is the similarity metric we will use

_, accuracy = predict_labels(training_set, training_set_labels, test_set, test_set_labels, global_alignment, check_genre_match, 20)

print("ACCURACY: ", accuracy)







# # Load the pre-trained Word2Vec model
# model_path = 'GoogleNews-vectors-negative300-SLIM.bin'  # Update with the correct path
# word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)

# # Example: Access vector for a word
# vector = word_vectors['king']
# similar_words = word_vectors.most_similar('king')
# print(similar_words)


# similarity = word_vectors.similarity('water', 'water')
# print(f"Similarity between 'king' and 'queen': {similarity}")

# if "KINg" not in word_vectors:
#     print("DOES NOT EXIST")