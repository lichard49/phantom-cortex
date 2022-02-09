from sklearn.ensemble import RandomForestClassifier
from classifying import get_test_train, get_windows, fit_model
from label import import_json, load_data
from analytics import plot_confusion_matrix

from brainflow import BoardShim
import numpy as np

d = import_json("files.json")
file_loc = "toma-ssvep-11-09-21/"

sampling_rate = BoardShim.get_sampling_rate(0) # change based on board
window_len = 2 # seconds

window_len *= sampling_rate

data = []
labels = []

for f in d:
    print(file_loc + f)
    readdata = load_data(file_loc + f)
    data.append(get_windows(readdata[1:9], window_len))
    labels.append(d[f][1]) # reading just the "Hz"

print(len(data))

# add FFT as features before testing

# test_X, test_Y, train_X, train_Y = get_test_train(data, labels)

# clf = RandomForestClassifier(random_state=0)
# fit_model(clf, train_X, train_Y, filename='ssvep.pkl')

# plot_confusion_matrix(clf, test_X, test_Y)