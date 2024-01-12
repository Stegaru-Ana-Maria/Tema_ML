import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from math import log
import matplotlib.pyplot as plt

main_directory = "../Tema/lingspam_public/lingspam_public"
subdirectories = ["bare", "lemm", "lemm_stop", "stop"]

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.isalpha() and word not in stop_words]

def train_naive_bayes(X, y):
    class_word_counts = defaultdict(Counter)
    class_counts = Counter()
    vocabulary = set()

    for i in range(len(X)):
        class_word_counts[y[i]] += Counter(X[i])
        class_counts[y[i]] += len(X[i])
        vocabulary.update(X[i])

    return class_word_counts, class_counts, vocabulary

def predict_naive_bayes(class_word_counts, class_counts, vocabulary, document):
    best_class = None
    max_log_prob = float('-inf')

    for c in class_counts:
        log_prob = log(class_counts[c] / sum(class_counts.values()))

        for word in document:
            log_prob += log((class_word_counts[c][word] + 1) / (class_counts[c] + len(vocabulary)))

        if log_prob > max_log_prob:
            max_log_prob = log_prob
            best_class = c

    return best_class

def evaluate_naive_bayes(class_word_counts, class_counts, vocabulary, X_test, y_test):
    correct_predictions = sum(predict_naive_bayes(class_word_counts, class_counts, vocabulary, X_test[i]) == y_test[i] for i in range(len(X_test)))
    accuracy = correct_predictions / len(X_test)
    return accuracy

def loocv(X, y):
    accuracies = []

    for i in range(len(X)):
        X_train = X[:i] + X[i+1:]
        y_train = y[:i] + y[i+1:]
        X_test = [X[i]]
        y_test = [y[i]]

        class_word_counts, class_counts, vocabulary = train_naive_bayes(X_train, y_train)
        accuracy = evaluate_naive_bayes(class_word_counts, class_counts, vocabulary, X_test, y_test)
        accuracies.append(accuracy)

    return accuracies

def process_file(file_path, X, y):
    try:
        with open(file_path, "r", encoding="latin-1") as file:
            content = file.read()
            preprocessed_content = preprocess_text(content)
            X.append(preprocessed_content)
            y.append(1 if file_path.endswith("spmsg") else 0)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

X_train, y_train, X_test, y_test = [], [], [], []

for subdir in subdirectories:
    subdir_path = os.path.join(main_directory, subdir)
    for part in range(1, 11):
        part_path = os.path.join(subdir_path, f"part{part}")
        for filename in os.listdir(part_path):
            file_path = os.path.join(part_path, filename)

            if part <= 9:
                process_file(file_path, X_train, y_train)
            else:
                process_file(file_path, X_test, y_test)

accuracy_results = loocv(X_train, y_train)

plt.figure(figsize=(10, 6))
plt.plot(accuracy_results, marker='o')
plt.xlabel('Iterație LOOCV')
plt.ylabel('Acuratețe')
plt.title('Rezultate LOOCV pentru Naive Bayes')
plt.show()
