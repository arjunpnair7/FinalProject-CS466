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

# Parameters for global alignment
GAP = "-"
SIMILARITY_THRESHOLD = 1
SIMILARITY_REWARD = 10
GAP_PENALTY = -1
STRING_MISMATCH_PENALTY = -5


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