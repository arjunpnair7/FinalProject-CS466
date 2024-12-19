from gensim.models import KeyedVectors

FILE_PATH = 'imdb_top_1000.csv'

model_path = 'GoogleNews-vectors-negative300-SLIM.bin'
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
word_set = set(word_vectors.key_to_index.keys())

vector = word_vectors['king']
similar_words = word_vectors.most_similar('king')
print(similar_words, "\n")
print(word_vectors.most_similar('water'), "\n")
print(word_vectors.most_similar('animal'), "\n")
print("---Similarities---")
print("(king, water)", word_vectors.similarity('king', 'water'))
print("(king, sultan)", word_vectors.similarity('king', 'sultan'))
print("(dog, water)", word_vectors.similarity('dog', 'water'))
print("(king, water)", word_vectors.similarity('king', 'water'))