import sklearn
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import numpy as np

# divides data into windows
# params: data in the dimensions of ch, samples
# output:
def get_windows(data, window_size):
    # iterate through each channel
    for ch_data in data:
        ch_windowed = []
        for i in np.arange(0, len(data[0])//window_size):
            ch_windowed.append(ch_data[i*window_size:(i*window_size)+window_size])
        ch_data = ch_windowed # replace singel channel data with windowed data 
    return data

# splits data into test and train sets
# params: windowed data, corresponding labels, test_prop is optional proportion
# output: test and train (window) data with corresponding labels
def get_test_train(data, labels, test_prop=0.1):

    # determine number of test and train sets
    test_num = round(len(data) * test_prop) # round to integer number of sets
    if test_num == 0: # test_num cannot be zero
        test_num = 1
    train_num = len(data) - test_num

    # randomly select test sets
    test_indicies = np.random.rand(len(data), size=test_num)

    # return variables / X and Y same length
    test_X = []
    test_Y = []
    train_X = []
    train_Y = []

    # data and labels should be the same dimesions (at the highest level)
    for set_index, dataset in enumerate(data):
        if set_index in test_indicies:
            for window in dataset:
                test_X.append(window)
                test_Y.append(labels[set_index])
        else:
            for window in dataset:
                train_X.append(window)
                train_Y.append(labels[set_index])
    
    return test_X, test_Y, train_X, train_Y
        
# fits model based on training data and write model to file
# params: 
# output: 
def fit_model(model, train_X, train_Y, filename='model.pkl'):
    # model = RandomForestClassifier(random_state=0)
    model.fit(train_X, train_Y)
    # write model to file
    joblib.dump(model, filename)

# loads saved model from .pkl file
# params: filename of the saved model
# output: model from saved file
def load_model(filename='model.pkl'):
    return joblib.load(filename)

# simple spectral analysis using max value of FFT
# params: frequency domain data, range and labels for each value, range of frequency to inspect
# output: highest band
def spectral_analysis(fft, value, value_labels, range=(10, 31)):
    # value = [(10, 14), (15, 19), (20, 24), (25, 29)]
    # value_label = [12, 17, 22, 27]
    freq = max(fft[range[0]:range[1]])
    for i, b in enumerate(value):
        if freq >= b[0] or freq <= b[1]:
            freq = value_labels[i]
    return freq