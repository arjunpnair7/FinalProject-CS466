import unittest
import global_alignment
from Utils import GAP, GAP_PENALTY, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY


# Unit test class
class TestGlobalAlignmentAlgorithm(unittest.TestCase):
    
    # Test to see if the custom reward function that compares the cosine similarity is working
    # These comparisons were taken from the similarities.txt
    def test_string_alignment_reward(self):
        self.assertEqual(global_alignment.string_alignment_reward_function("king", "queen"), SIMILARITY_REWARD)
        self.assertEqual(global_alignment.string_alignment_reward_function("king", "sultan"), SIMILARITY_REWARD)
        self.assertEqual(global_alignment.string_alignment_reward_function("animals", "dog"), SIMILARITY_REWARD)
        self.assertEqual(global_alignment.string_alignment_reward_function("animals", "cat"), SIMILARITY_REWARD)

        self.assertEqual(global_alignment.string_alignment_reward_function(GAP, "dog"), GAP_PENALTY)
        self.assertEqual(global_alignment.string_alignment_reward_function("dog", GAP), GAP_PENALTY)

        self.assertEqual(global_alignment.string_alignment_reward_function("king", "water"), STRING_MISMATCH_PENALTY)
        self.assertEqual(global_alignment.string_alignment_reward_function("dog", "water"), STRING_MISMATCH_PENALTY)
        self.assertEqual(global_alignment.string_alignment_reward_function("king", "water"), STRING_MISMATCH_PENALTY)



    # Tests to see if the global alignment algorithm is functioning as expected
    # I used a well-known biology-based reward function so I can test the actual logic of the algorithm itself
    # By using a well known reward function, I could check for the accuracy easily. I used the following resource from class to generate the correct score values: http://rna.informatik.uni-freiburg.de/Teaching/index.jsp?toolName=Needleman-Wunsch
    def test_complex_global_alignment(self):
        self.assertEqual(global_alignment.global_alignment("CDGAX", "ACAGGX", use_default_reward=True), 0)
        self.assertEqual(global_alignment.global_alignment("ACTG", "ACTG", use_default_reward=True), 4)  # Identical sequences
        self.assertEqual(global_alignment.global_alignment("AAAA", "TTTT", use_default_reward=True), -4)  # Completely different sequences
        self.assertEqual(global_alignment.global_alignment("ACTG", "AC", use_default_reward=True), 0)  # Partial match at the start
        self.assertEqual(global_alignment.global_alignment("ACTG", "", use_default_reward=True), -4)  # One empty sequence
        self.assertEqual(global_alignment.global_alignment("", "", use_default_reward=True), 0)  # Both sequences empty
        self.assertEqual(global_alignment.global_alignment("GATTACA", "GCATGCU", use_default_reward=True), 0)  # Example with mismatches and gaps
        self.assertEqual(global_alignment.global_alignment("AAAAA", "AAAA", use_default_reward=True), 3)  # One sequence longer than the other
        self.assertEqual(global_alignment.global_alignment("AGTACGCA", "TATGC", use_default_reward=True), 0)  # Example with alignment gaps
        self.assertEqual(global_alignment.global_alignment("A", "G", use_default_reward=True), -1)  # Single characters mismatched


if __name__ == "__main__":
    unittest.main()
