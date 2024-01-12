import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def euclidean_distance(instance1, instance2):
    features = set(instance1.keys()) | set(instance2.keys())
    distance = np.sqrt(sum((instance1.get(feature, 0) - instance2.get(feature, 0)) ** 2 for feature in features))
    return distance

def get_neighbors(X_train, y_train, test_instance, k):
    distances = [euclidean_distance(train_instance, test_instance) for train_instance in X_train]
    indices = np.argsort(distances)[:k]
    neighbors = [y_train[i] for i in indices]
    return neighbors

def majority_vote(neighbors):
    counter = Counter(neighbors)
    return counter.most_common(1)[0][0]

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_instance in X_test:
        print("Calculating distances for test instance:", test_instance)
        neighbors = get_neighbors(X_train, y_train, test_instance, k)
        print("Neighbors:", neighbors)
        predicted_label = majority_vote(neighbors)
        print("Predicted label:", predicted_label)
        predictions.append(predicted_label)
    return predictions

main_directory = "../Tema/lingspam_public/lingspam_public"

subdirectories = ["bare", "lemm", "lemm_stop", "stop"]

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
                        X_train.append(Counter(preprocessed_content))
                        y_train.append(1 if filename.startswith("spmsg") else 0)
                    else:
                        X_test.append(Counter(preprocessed_content))
                        y_test.append(1 if filename.startswith("spmsg") else 0)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

k_value = 5
predictions = knn_predict(X_train, y_train, X_test, k_value)
print("Predictions:", predictions)

accuracy = sum(1 for true, pred in zip(y_test, predictions) if true == pred) / len(y_test)
print(f'Acuratete pe setul de testare: {accuracy}')
