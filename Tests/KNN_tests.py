import unittest
from collections import Counter
from Models.knn import predict_labels

# Similarity function: negative Manhattan distance
def similarity_function(x, y):
    return -sum(abs(a - b) for a, b in zip(x, y))

# Evaluation function: exact match
def evaluation_function(predicted_label, true_label):
    return 1 if predicted_label == true_label else 0


# Unit test class
class TestKNN(unittest.TestCase):

    def test_single_training_point(self):
        training_set = [[1, 1]]
        training_labels = [0]
        test_set = [[1, 1], [2, 2]]
        test_labels = [0, 0]
        predicted_labels, accuracy = predict_labels(
            training_set, training_labels, test_set, test_labels, similarity_function, evaluation_function, K=1
        )
        self.assertEqual(predicted_labels, [0, 0])
        self.assertEqual(accuracy, 1.0)

    def test_two_points_one_neighbor(self):
        training_set = [[1, 1], [2, 2]]
        training_labels = [0, 1]
        test_set = [[1, 1], [2, 2], [1.5, 1.5]]
        test_labels = [0, 1, 0]
        predicted_labels, accuracy = predict_labels(
            training_set, training_labels, test_set, test_labels, similarity_function, evaluation_function, K=1
        )
        self.assertEqual(predicted_labels, [0, 1, 0])
        self.assertEqual(accuracy, 1.0)

    def test_tie_in_labels(self):
        training_set = [[1, 1], [2, 2], [1, 0]]
        training_labels = [0, 1, 0]
        test_set = [[1.5, 1.5]]
        test_labels = [0]
        predicted_labels, accuracy = predict_labels(
            training_set, training_labels, test_set, test_labels, similarity_function, evaluation_function, K=2
        )
        self.assertEqual(predicted_labels, [0])
        self.assertEqual(accuracy, 1.0)

    def test_custom_similarity_and_evaluation(self):
        training_set = [[0, 0], [1, 1]]
        training_labels = [1, 0]
        test_set = [[0.5, 0.5]]
        test_labels = [1]
        predicted_labels, accuracy = predict_labels(
            training_set, training_labels, test_set, test_labels, similarity_function, evaluation_function, K=2
        )
        self.assertEqual(predicted_labels, [1])
        self.assertEqual(accuracy, 1.0)

    def test_low_accuracy_case(self):
        training_set = [[1, 1], [3, 3], [2, 2]]
        training_labels = [0, 1, 0]
        test_set = [[1, 1], [2, 2], [3, 3]]
        test_labels = [1, 0, 0]  # Deliberately mismatched
        predicted_labels, accuracy = predict_labels(
            training_set, training_labels, test_set, test_labels, similarity_function, evaluation_function, K=1
        )
        self.assertEqual(predicted_labels, [0, 0, 1])
        self.assertAlmostEqual(accuracy, 1 / 3)


if __name__ == "__main__":
    unittest.main()
