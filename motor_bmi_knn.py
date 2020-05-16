import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm   # Shows a progress bar to know the progress during training.

def read_file(file_name):
    lines = [line.rstrip('\n').split('\t') for line in open(file_name)]
    file_np = np.array(lines).astype(np.float)
    if file_np.shape[0] == 1:
        file_np = file_np.flatten()
    return file_np

# Read training files
angle_train = read_file('Angle_Training.txt')
spike_train = read_file('Training_SpikeTrains.txt')
# Give labels to the 4 angle classes
train_labels = np.zeros(angle_train.shape[0])
train_labels[np.where(np.logical_and(angle_train >= 90, angle_train < 180))] = 1
train_labels[np.where(np.logical_and(angle_train >= 180, angle_train < 270))] = 2
train_labels[np.where(np.logical_and(angle_train >= 270, angle_train < 360))] = 3

# Read test files
angle_test = read_file('Angle_Testing.txt')
spike_test = read_file('Testing_SpikeTrains.txt')
# Give labels to the 4 angle classes
test_labels = np.zeros(angle_test.shape[0])
test_labels[np.where(np.logical_and(angle_test >= 90, angle_test < 180))] = 1
test_labels[np.where(np.logical_and(angle_test >= 180, angle_test < 270))] = 2
test_labels[np.where(np.logical_and(angle_test >= 270, angle_test < 360))] = 3

all_acc = []
k_start = 1
k_finish = 301
for i in tqdm(range(k_start,k_finish)):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(spike_train.T, train_labels)
    preds = neigh.predict(spike_test.T)
    acc = accuracy_score(test_labels, preds) * 100
    all_acc.append(acc)

print('Best Accuracy: ' + str(np.max(all_acc)) + '%')
print('At K = ' + str(np.argmax(all_acc) + 1))

plt.figure()
plt.plot(np.arange(k_start,k_finish), np.array(all_acc))
plt.xlabel('K-value')
plt.ylabel('Accuracy %')
plt.show()