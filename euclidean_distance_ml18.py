import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to the value less then total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    # Counter(votes) creates a dictionary with number of times each element occurs in the list.
    confidence = Counter(votes).most_common(1)[0][1] / k  # first element is actually how many
    # print(vote_result, confidence)

    return vote_result, confidence


accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    # to be sure that I have correct data type, I have to convert my data set to float (or sometimes int, but mostly float)
    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)  # shuffle the data

    test_size = 0.4
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]  # everything up to the last 20%
    test_data = full_data[-int(test_size * len(full_data)):]  # last 20% of the data

    for i in train_data:
        # -1 in the train_set is the 'class' column in the data set ---> value is 2 or 4 --> (2 for benign, 4 for malignant)
        train_set[i[-1]].append(i[:-1])
        # we are getting the class value and appending it to the list train_set all up to the last value from train_data

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data,
                                                   k=5)  # increasing the K does not give me better accuracy
            if group == vote:
                correct += 1
            total += 1

    # print('Accuracy:', correct / total)
    accuracies.append(correct / total)

print(sum(accuracies) / len(accuracies))
# accuracy - did we get the classification right
# confidence - when classifier says "we have 100% of the votes in favor of this class being such as such"
