from gensim.models import KeyedVectors
from Utils import FILE_PATH, word_vectors


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