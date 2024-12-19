import pandas as pd
from knn import KNN
from gensim.models import KeyedVectors


FILE_PATH = 'imdb_top_1000.csv'
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by',
    'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not',
    'of', 'on', 'or', 'such', 'that', 'the', 'their',
    'then', 'there', 'these', 'they', 'this',
    'to', 'was', 'will', 'with'
}
STOP_WORDS = {}
DATASET_SIZE = 1000
TRAININGSET_SIZE = 900


model_path = 'GoogleNews-vectors-negative300-SLIM.bin'
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
word_set = set(word_vectors.key_to_index.keys())



class MovieEntry:
    def __init__(self, overview, genre):
        # Filter out stop words and words not in word_set during initialization
        self.overview = [word for word in overview.lower().split() if word not in STOP_WORDS and word in word_set]
        self.genre = [g.strip().lower() for g in genre.split(',')]

    def __str__(self):
        snippet = ' '.join(self.overview[:10]) + '...' if len(self.overview) > 10 else ' '.join(self.overview)
        return f'Genre: {self.genre}, Overview: {snippet}'

df = pd.read_csv(FILE_PATH)
movie_entries = [MovieEntry(row['Overview'], row['Genre']) for _, row in df.iterrows()]



training_set = movie_entries[:900]
test_set = movie_entries[900: 925]

knn = KNN(training_set, test_set)
knn.evaluate_performance()

# for entry in movie_entries:
#     print(entry)


# print(len(movie_entries))





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