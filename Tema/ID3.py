import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from math import log2

main_directory = "../Tema/lingspam_public/lingspam_public"

subdirectories = ["bare", "lemm", "lemm_stop", "stop"]

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def entropy(labels):
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1

    entropy_val = 0.0
    total_samples = len(labels)

    for count in label_counts.values():
        prob = count / total_samples
        entropy_val -= prob * log2(prob)

    return entropy_val

def information_gain(data, labels, feature_index):
    total_entropy = entropy(labels)

    if len(data) == 0 or feature_index >= len(data):
        return 0.0

    feature_values = set(data[feature_index][i] for i in range(len(data[feature_index])))
    weighted_entropy = 0.0

    for value in feature_values:
        subset_indices = [i for i in range(len(data)) if len(data[i]) > feature_index and data[i][feature_index][0] == value]
        subset_labels = [labels[i] for i in subset_indices]
        weighted_entropy += (len(subset_labels) / len(labels)) * entropy(subset_labels)

    return total_entropy - weighted_entropy

def choose_best_feature(data, labels):
    num_features = len(data[0])
    gains = [information_gain(data, labels, i) for i in range(num_features)]
    best_feature_index = gains.index(max(gains))
    return best_feature_index

class TreeNode:
    def __init__(self, value=None, decision=None):
        self.value = value
        self.decision = decision
        self.children = {}

def build_decision_tree(data, labels, feature_names):
    if len(set(labels)) == 1:
        return TreeNode(decision=labels[0])

    if len(feature_names) == 0:
        return TreeNode(decision=max(set(labels), key=labels.count))

    best_feature_index = choose_best_feature(data, labels)
    best_feature_name = feature_names[best_feature_index]

    sub_tree = TreeNode(value=best_feature_name)

    feature_values = set(data[i][best_feature_index] for i in range(len(data)))
    for value in feature_values:
        subset_indices = [i for i in range(len(data)) if data[i][best_feature_index] == value]
        subset_data = [data[i][:best_feature_index] + data[i][best_feature_index + 1:] for i in subset_indices]
        subset_labels = [labels[i] for i in subset_indices]

        if not subset_data:
            sub_tree.children[value] = TreeNode(decision=max(set(labels), key=labels.count))
        else:
            sub_tree.children[value] = build_decision_tree(subset_data, subset_labels, feature_names[:best_feature_index] + feature_names[best_feature_index + 1:])

    return sub_tree

def predict(tree, sample):
    if tree.decision is not None:
        return tree.decision
    if tree.value is not None and tree.value in feature_names and sample[feature_names.index(tree.value)] in tree.children:
        return predict(tree.children[sample[feature_names.index(tree.value)]], sample)
    return None

X_train = []
y_train = []

X_test = []
y_test = []

for subdir in subdirectories:
    subdir_path = os.path.join(main_directory, subdir)

    for part in range(1, 11):
        part_path = os.path.join(subdir_path, f"part{part}")

        for filename in os.listdir(part_path):
            file_path = os.path.join(part_path, filename)

            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    content = file.read()

                    preprocessed_content = preprocess_text(content)

                    if part <= 9:
                        X_train.append(preprocessed_content)
                        y_train.append(1 if filename.startswith("spmsg") else 0)
                    else:
                        X_test.append(preprocessed_content)
                        y_test.append(1 if filename.startswith("spmsg") else 0)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


feature_names = [f"feature_{i}" for i in range(len(X_train[0]))]

decision_tree = build_decision_tree(X_train, y_train, feature_names)

correct_predictions = 0
for i in range(len(X_test)):
    prediction = predict(decision_tree, X_test[i])
    if prediction is not None and prediction == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
print(f"Acuratețe pe setul de testare: {accuracy}")
