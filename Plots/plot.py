import matplotlib.pyplot as plt
from Utils import GAP, GAP_PENALTY, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY, SIMILARITY_THRESHOLD


x1, y1 = [10, 20, 30, 40, 50, 60, 70], [.68, .64, .68, .64, .68, .72, .64]
x2, y2 = [.2, .4, .6, .8, 1], [0.84, 0.64, 0.68, 0.68, 0.68]
x3, y3 = [-.25, -.5, -.75, -1, -1.25, -1.50, -1.75], [0.64, 0.68, 0.68, 0.64, 0.68, 0.68, 0.68]

GAP_PENALTY, SIMILARITY_THRESHOLD, SIMILARITY_REWARD, STRING_MISMATCH_PENALTY = -1, 0.7, 1, -0.5

# K vs Accuracy
default_reward_string = f"Gap Penalty: {GAP_PENALTY}, Similarity Threshold: {SIMILARITY_THRESHOLD}, Similarity Reward: {SIMILARITY_REWARD}, String Mismatch Penalty: {STRING_MISMATCH_PENALTY}"

plt.figure(figsize=(8, 6))
plt.plot(x1, y1, marker='o', linestyle='-', color='b', label='Accuracy')
plt.title("KNN Performance vs K", fontsize=14)
plt.xlabel("K Values", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
# plt.tight_layout()
plt.savefig('K_value_vs_Accuracy.png')

# # Threshold vs Accuracy
threshold_string = f"Gap Penalty: {GAP_PENALTY}, K: {30}, Similarity Reward: {SIMILARITY_REWARD}, String Mismatch Penalty: {STRING_MISMATCH_PENALTY}"
plt.figure(figsize=(8, 6))
plt.plot(x2, y2, marker='s', linestyle='--', color='r', label='Accuracy')
plt.title("KNN Performance vs Similarity Threshold", fontsize=14)
plt.xlabel("Similarity Threshold", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

plt.savefig('Similarity_Threshold_vs_Accuracy.png')

# Mismatch Reward vs Accuracy
mismatch_string = f"Gap Penalty: {GAP_PENALTY}, Similarity Threshold: {SIMILARITY_THRESHOLD}, Similarity Reward: {SIMILARITY_REWARD}"
plt.figure(figsize=(8, 6))
plt.plot(x3, y3, marker='d', linestyle='-.', color='g', label='Accuracy')
plt.title("KNN Performance vs Mismatch Reward", fontsize=14)
plt.xlabel("Mismatch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('Mismatch_Reward_vs_Accuracy.png')

# Display updated plots
plt.show()

