import unittest
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from Models.global_alignment import global_alignment, string_alignment_reward_function
from Plots.Utils import GAP, GAP_PENALTY, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY, SIMILARITY_THRESHOLD

# GAP = "-"
# SIMILARITY_THRESHOLD = .60
# SIMILARITY_REWARD = 10
# GAP_PENALTY = -1
# STRING_MISMATCH_PENALTY = -.5

rewards = (GAP_PENALTY, SIMILARITY_THRESHOLD, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY)


# Unit test class
class TestGlobalAlignmentAlgorithm(unittest.TestCase):

    # Test to see if the custom reward function that compares the cosine similarity is working
    # These comparisons were taken from the similarities.txt
    def test_string_alignment_reward(self):
        self.assertEqual(string_alignment_reward_function("king", "queen", rewards), SIMILARITY_REWARD)
        self.assertEqual(string_alignment_reward_function("king", "sultan", rewards), STRING_MISMATCH_PENALTY)
        self.assertEqual(string_alignment_reward_function("animals", "dog", rewards), STRING_MISMATCH_PENALTY)
        self.assertEqual(string_alignment_reward_function("animals", "cat", rewards), STRING_MISMATCH_PENALTY)

        self.assertEqual(string_alignment_reward_function(GAP, "dog", rewards), GAP_PENALTY)
        self.assertEqual(string_alignment_reward_function("dog", GAP, rewards), GAP_PENALTY)

        self.assertEqual(string_alignment_reward_function("king", "water", rewards), STRING_MISMATCH_PENALTY)
        self.assertEqual(string_alignment_reward_function("dog", "water", rewards), STRING_MISMATCH_PENALTY)
        self.assertEqual(string_alignment_reward_function("king", "water", rewards), STRING_MISMATCH_PENALTY)



    # Tests to see if the global alignment algorithm is functioning as expected
    # I used a well-known biology-based reward function so I can test the actual logic of the algorithm itself
    # By using a well known reward function, I could check for the accuracy easily. I used the following resource from class to generate the correct score values: http://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Needleman-Wunsch
    def test_complex_global_alignment(self):
        self.assertEqual(global_alignment("CDGAX", "ACAGGX", rewards, use_default_reward=True), 0)
        self.assertEqual(global_alignment("ACTG", "ACTG", rewards, use_default_reward=True), 4)  # Identical sequences
        self.assertEqual(global_alignment("AAAA", "TTTT", rewards, use_default_reward=True), -4)  # Completely different sequences
        self.assertEqual(global_alignment("ACTG", "AC", rewards, use_default_reward=True), 0)  # Partial match at the start
        self.assertEqual(global_alignment("ACTG", "", rewards, use_default_reward=True), -4)  # One empty sequence
        self.assertEqual(global_alignment("", "", rewards, use_default_reward=True), 0)  # Both sequences empty
        self.assertEqual(global_alignment("GATTACA", "GCATGCU", rewards, use_default_reward=True), 0)  # Example with mismatches and gaps
        self.assertEqual(global_alignment("AAAAA", "AAAA", rewards, use_default_reward=True), 3)  # One sequence longer than the other
        self.assertEqual(global_alignment("AGTACGCA", "TATGC", rewards, use_default_reward=True), 0)  # Example with alignment gaps
        self.assertEqual(global_alignment("A", "G", rewards, use_default_reward=True), -1)  # Single characters mismatched


if __name__ == "__main__":
    unittest.main()
