from collections import Counter

def predict_labels(training_set, training_labels, test_set, test_labels, similarity_function, evaluation_function, K, rewards):


    def predict_single_label(test_point):
            scores = []
            for index, cur_entry in enumerate(training_set):
                similarity_score = similarity_function(test_point, cur_entry, rewards)
                scores.append((index, similarity_score))

            scores.sort(key=lambda x: x[1], reverse=True)
            top_k_indices = [score[0] for score in scores[:K]]
            top_k_labels = []
            for index in top_k_indices:
                top_k_labels.append(tuple(training_labels[index]))


            label_counts = Counter(tuple(top_k_labels))
            predicted_label = label_counts.most_common(1)[0][0]


            return predicted_label
    

    correct_count = 0
    total = len(test_set)
    predicted_label_list = []
    for idx in range(len(test_set)):
        predicted_label = predict_single_label(test_set[idx])
        predicted_label_list.append(predicted_label)

        true_label = test_labels[idx]
        correct_count += evaluation_function(predicted_label, true_label)

    return predicted_label_list, (correct_count / total)
    
    