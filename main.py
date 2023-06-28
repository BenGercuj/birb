import matplotlib.pyplot as plt
from librosa import load
from scipy.fft import fft
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def get_data(folders):
    label = 0
    training_td = []
    training_fd = []
    test_td = []
    test_fd = []
    training_y = []
    test_y = []
    y_name = []

    for i in folders:
        path = './data/' + i.name
        wav_files = list(Path(path).glob('*.mp3'))
        y_name.append(i.name)

        counter = 0
        for j in wav_files:
            counter += 1

            data_td, fs = load(j, sr=32000, mono=True, duration=8)
            data_fd = fft(data_td)

            if counter < 10:
                training_td.append(data_td)
                training_fd.append(abs(data_fd[0:len(data_fd) // 2]))
                training_y.append(label)

            else:
                test_td.append(data_td)
                test_fd.append(abs(data_fd[0:len(data_fd) // 2]))
                test_y.append(label)

        label += 1

    return training_td, training_fd, training_y, test_td, test_fd, test_y, y_name


# Getting data
birb_folders = list(Path('./data').glob('*'))
training_td, training_fd, training_y, test_td, test_fd, test_y, y_name = get_data(birb_folders)

# Classification
knn = KNeighborsClassifier(n_neighbors=5, metric='correlation')
knn.fit(training_td, training_y)
print("Time-domain accuracy: ", knn.score(test_td, test_y) * 100, "%")

knn.fit(training_fd, training_y)
print("Frequency-domain accuracy: ", knn.score(test_fd, test_y) * 100, "%")

# Plotting TD and FD of training dataset for documentation
# previous_class = training_y[0]
# for i in range(len(training_y)):
#     current_class = training_y[i]
#     if current_class == previous_class and i != len(training_y) - 1:
#         title = "Time-domain of all " + y_name[training_y[i]] + " birds within training dataset"
#         plt.title(title)
#         plt.plot(training_td[i])
#
#     else:
#         plt.show()
#
#     previous_class = current_class
#
# previous_class = training_y[0]
# for i in range(len(training_y)):
#     current_class = training_y[i]
#     if current_class == previous_class and i != len(training_y) - 1:
#         title = "Frequency-domain of all " + y_name[training_y[i]] + " birds within training dataset"
#         plt.title(title)
#         plt.plot(training_fd[i])
#
#     else:
#         plt.show()
#
#     previous_class = current_class

# Adding noise to test dataset
noise_data = []
accuracy = []
knn.fit(training_td, training_y)

for i in np.arange(0, 1, 0.1):
    noise = np.random.normal(0, i, len(test_td[0]))
    test_td_noise = test_td[:] + noise

    print("Time-domain accuracy with noise: ", knn.score(test_td_noise, test_y) * 100, "%")
    noise_data.append(i)
    accuracy.append(knn.score(test_td_noise, test_y) * 100)

plt.title("Classification accuracy given noise")
plt.plot(noise_data, accuracy)
plt.xlabel("Noise deviation")
plt.ylabel("Accuracy (%)")
plt.show()