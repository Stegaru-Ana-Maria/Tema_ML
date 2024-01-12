import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    for iteration in range(max_iters):
        labels = np.argmin(np.array([[calculate_distance(x, centroid) for centroid in centroids] for x in X]), axis=1)

        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        if np.all(centroids == new_centroids):
            print(f"Converged after {iteration + 1} iterations.")
            break

        centroids = new_centroids
        print(f"Iteration {iteration + 1}: Updated centroids.")

    return labels

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
                        X_train.append(preprocessed_content)
                        y_train.append(1 if filename.startswith("spmsg") else 0)
                    else:
                        X_test.append(preprocessed_content)
                        y_test.append(1 if filename.startswith("spmsg") else 0)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


vocab_size = 1000
word_to_index = {word: idx for idx, word in enumerate(np.unique(np.concatenate(X_train)))}
X_train_vectors = np.zeros((len(X_train), vocab_size))

for i, doc in enumerate(X_train):
    for word in doc:
        if word in word_to_index and word_to_index[word] < vocab_size:
            X_train_vectors[i, word_to_index[word]] += 1

num_clusters = 2

print("Training K-means on the training set...")
train_cluster_labels = kmeans(X_train_vectors, num_clusters)

X_test_vectors = np.zeros((len(X_test), vocab_size))

for i, doc in enumerate(X_test):
    for word in doc:
        if word in word_to_index and word_to_index[word] < vocab_size:
            X_test_vectors[i, word_to_index[word]] += 1

print("Applying K-means on the test set...")
test_cluster_labels = kmeans(X_test_vectors, num_clusters)

correct_predictions = sum(1 for label, true_label in zip(test_cluster_labels, y_test) if label == true_label)
accuracy = correct_predictions / len(y_test)
print(f'Acuratete pe setul de testare: {accuracy}')
