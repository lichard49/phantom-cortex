from sklearn.ensemble import RandomForestClassifier
from classifying import get_test_train, get_windows, fit_model
from processing import get_fft_ch
from label import import_json, load_data
from analytics import plot_confusion_matrix

from brainflow import BoardShim
import numpy as np
import matplotlib.pyplot as plt

#### old

# d = import_json("files.json")
# file_loc = "toma-ssvep-11-09-21/"

sampling_rate = BoardShim.get_sampling_rate(0) # change based on board
window_len = 2 # seconds

window_len *= sampling_rate

# filenames = []
# data = []
# labels = []

# print(d)

# for f in d:
#     print(file_loc + f)
#     filenames.append(f)
#     readdata = load_data(file_loc + f)
#     windowed = get_windows(readdata[1:9], window_len)
#     print("windowed: ch x windows x wind_len"+str(np.shape(windowed)))
#     ch_data = []
#     for ch in windowed:
#         fft_ch = [] #stores fft for single channel (multiple windows)
#         for ch_wd in ch:
#             fft_ch_wd = get_fft_ch(ch_wd, sampling_rate) # choosing seventh channel
#             fft_ch.append(fft_ch_wd)
#         ch_data.append(fft_ch)
#     print(np.shape(ch_data))
#     data.append(np.concatenate(ch_data)) # window x FFT for each file
#     labels.append(d[f][1]) # reading just the "Hz"

# print(filenames)

# print(np.shape(data))
# print(data[0])

# plt.plot(np.arange(len(data[0])), data[0])
# plt.show()

# add FFT as features before testing

### new

train_X, train_Y, test_X, test_Y = get_test_train("test_train_1.json", "files.json", "/Users/tomaitagaki/Documents/GitHub/data/toma-ssvep-11-09-21/", window_len)

test_X = np.array(test_X)
test_Y = np.array(test_Y)
train_X = np.array(train_X)
train_Y = np.array(train_Y)

plt.figure(train_X[0])

print(test_X.shape, train_X.shape, test_Y.shape)

clf = RandomForestClassifier(random_state=0)
fit_model(clf, train_X, train_Y, filename='ssvep.pkl')

plot_confusion_matrix(clf, test_X, test_Y)