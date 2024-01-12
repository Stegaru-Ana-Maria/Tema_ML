import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import pandas as pd

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


def stump_predict(stump, sample):
    feature, threshold, polarity = stump
    value = sample.get(feature, 0)
    return 1 if (value * polarity) < (threshold * polarity) else -1

def build_stump(X, y, weights):
    m, n = X.shape
    min_error = float('inf')
    best_stump = None

    for i in range(n):
        feature_values = set(np.unique(X[:, i]))
        for value in feature_values:
            for polarity in [1, -1]:
                threshold = value * polarity

                predictions = np.array(
                    [stump_predict((i, threshold, polarity), dict(zip(range(n), sample))) for sample in X])

                weighted_error = np.sum(weights * (predictions != y))

                print(f"Feature {i}, Threshold {threshold}, Polarity {polarity}, Weighted Error {weighted_error}")

                if weighted_error < min_error:
                    min_error = weighted_error
                    best_stump = (i, threshold, polarity)

    return best_stump, min_error


def update_weights(weights, alpha, predictions, y):
    weights *= np.exp(-alpha * y * predictions)
    weights /= np.sum(weights)
    return weights


def adaboost(X, y, num_classifiers):
    m, n = X.shape
    classifiers = []
    alphas = []
    weights = np.ones(m) / m

    for t in range(1, num_classifiers + 1):
        print(f"Training classifier {t}/{num_classifiers}")
        stump, error = build_stump(X, y, weights)
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
        predictions = np.array([stump_predict(stump, sample) for sample in X])

        print(f"Alpha {alpha}, Error {error}, Predictions {predictions}, Weights {weights}")

        weights = update_weights(weights, alpha, predictions, y)

        classifiers.append(stump)
        alphas.append(alpha)

    return classifiers, alphas


def adaboost_predict(classifiers, alphas, sample):
    predictions = [stump_predict((i, threshold, polarity), sample) for i, threshold, polarity in classifiers]
    return np.sign(np.dot(alphas, predictions))


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
                        y_train.append(1 if filename.startswith("spmsg") else -1)
                    else:
                        X_test.append(Counter(preprocessed_content))
                        y_test.append(1 if filename.startswith("spmsg") else -1)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

X_train = X_train_df.to_numpy()
X_test = X_test_df.to_numpy()
num_classifiers = 50

classifiers, alphas = adaboost(X_train, y_train, num_classifiers)

correct_predictions = sum(
    1 for sample, true_label in zip(X_test, y_test) if adaboost_predict(classifiers, alphas, sample) == true_label)
accuracy = correct_predictions / len(X_test)
print(f'Acuratete pe setul de testare: {accuracy}')
