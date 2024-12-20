from gensim.models import KeyedVectors
from Plots.Utils import GAP, word_vectors

cache = {}

def default_reward_function(char1,char2):
    if char1 == "-" or char2 == "-":
        return -1
    if char1 == char2:
        return 1
    else:
        return -1

def string_alignment_reward_function(word1, word2, rewards):
    GAP_PENALTY, SIMILARITY_THRESHOLD, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY = rewards
    if word1 == GAP or word2 == GAP:
        return GAP_PENALTY
    
    if (word1, word2) in cache:
        similarity = cache[(word1, word2)]
    else:
        similarity = word_vectors.similarity(word1, word2)
        cache[(word1, word2)] = similarity

    if similarity >= SIMILARITY_THRESHOLD:
        return SIMILARITY_REWARD
    else:
        return STRING_MISMATCH_PENALTY


def global_alignment(overview_1, overview_2, rewards, use_default_reward=False):
    if use_default_reward:
        rows, cols = len(overview_1) + 1, len(overview_2) + 1
        dp = [[0 for _ in range(len(overview_2) + 1)] for _ in range(len(overview_1) + 1)]
    else:
        rows, cols = len(overview_1.overview) + 1, len(overview_2.overview) + 1
        dp = [[0 for _ in range(len(overview_2.overview) + 1)] for _ in range(len(overview_1.overview) + 1)]

    GAP_PENALTY = rewards[0]

    if use_default_reward:
        for i in range(len(overview_1) + 1):
            dp[i][0] = i * GAP_PENALTY 
        for j in range(len(overview_2) + 1):
            dp[0][j] = j * GAP_PENALTY
    else:
        for i in range(len(overview_1.overview) + 1):
            dp[i][0] = i * GAP_PENALTY 
        for j in range(len(overview_2.overview) + 1):
            dp[0][j] = j * GAP_PENALTY 

    # Fill DP table
    
    for i in range(1, rows):
        for j in range(1, cols):
            if use_default_reward:
                match_score = dp[i - 1][j - 1] + default_reward_function(overview_1[i - 1], overview_2[j - 1])
                delete_score = dp[i - 1][j] + default_reward_function(overview_1[i - 1], GAP)
                insert_score = dp[i][j - 1] + default_reward_function(GAP, overview_2[j - 1])
            else:
                match_score = dp[i - 1][j - 1] + string_alignment_reward_function(overview_1.overview[i - 1], overview_2.overview[j - 1], rewards)
                delete_score = dp[i - 1][j] + string_alignment_reward_function(overview_1.overview[i - 1], GAP, rewards)
                insert_score = dp[i][j - 1] + string_alignment_reward_function(GAP, overview_2.overview[j - 1], rewards)

            dp[i][j] = max(match_score, delete_score, insert_score)

    return dp[rows - 1][cols - 1]


          
