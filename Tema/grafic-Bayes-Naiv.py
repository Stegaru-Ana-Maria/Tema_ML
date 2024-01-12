import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from math import log
import numpy as np
import matplotlib.pyplot as plt

main_directory = "../Tema/lingspam_public/lingspam_public"
subdirectories = ["bare", "lemm", "lemm_stop", "stop"]

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def train_naive_bayes(X, y):
    class_word_counts = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)
    vocabulary = set()

    for i in range(len(X)):
        for word in X[i]:
            class_word_counts[y[i]][word] += 1
            class_counts[y[i]] += 1
            vocabulary.add(word)

    return class_word_counts, class_counts, vocabulary

def predict_naive_bayes(class_word_counts, class_counts, vocabulary, document):
    best_class = None
    max_log_prob = float('-inf')

    for c in class_counts.keys():
        log_prob = log(class_counts[c] / sum(class_counts.values()))

        for word in document:
            log_prob += log((class_word_counts[c][word] + 1) / (class_counts[c] + len(vocabulary)))

        if log_prob > max_log_prob:
            max_log_prob = log_prob
            best_class = c

    return best_class

def evaluate_naive_bayes(class_word_counts, class_counts, vocabulary, X, y):
    correct_predictions = 0

    for i in range(len(X)):
        predicted_class = predict_naive_bayes(class_word_counts, class_counts, vocabulary, X[i])
        if predicted_class == y[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(X)
    return accuracy

train_accuracies = []
test_accuracies = []

for subdir in subdirectories:
    subdir_path = os.path.join(main_directory, subdir)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

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

    class_word_counts, class_counts, vocabulary = train_naive_bayes(X_train, y_train)

    train_accuracy = evaluate_naive_bayes(class_word_counts, class_counts, vocabulary, X_train, y_train)
    train_accuracies.append(train_accuracy)

    test_accuracy = evaluate_naive_bayes(class_word_counts, class_counts, vocabulary, X_test, y_test)
    test_accuracies.append(test_accuracy)

    print(f'Acuratetea pe setul de antrenare pentru {subdir}: {train_accuracy}')
    print(f'Acuratetea pe setul de testare pentru {subdir}: {test_accuracy}')

barWidth = 0.35
r1 = np.arange(len(train_accuracies))
r2 = [x + barWidth for x in r1]

plt.bar(r1, train_accuracies, color='green', width=barWidth, edgecolor='grey', label='Train Accuracy')
plt.bar(r2, test_accuracies, color='red', width=barWidth, edgecolor='grey', label='Test Accuracy')

plt.xlabel('Subdirectoare')
plt.ylabel('Acuratete')
plt.title('Acuratete pe Setul de Antrenare și Testare')
plt.xticks([r + barWidth/2 for r in range(len(train_accuracies))], subdirectories)
plt.legend()
plt.show()
