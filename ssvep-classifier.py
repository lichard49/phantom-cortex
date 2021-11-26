import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import numpy as np
import pandas as pd
from brainflow import DataFilter, board_shim, WindowFunctions
import matplotlib.pyplot as plt
import random
import collections

def load_data(filename):
    readdata = DataFilter.read_file(filename)
    return readdata

def get_fft(data):
    fs = 256
    fft = []
    for ch in data:
        nfft = DataFilter.get_nearest_power_of_two(fs)
        psd = DataFilter.get_psd_welch(ch, nfft, nfft // 2, fs,
                                WindowFunctions.BLACKMAN_HARRIS.value)
        fft.append(psd[0][:61])
    return fft

# take in data23 and returns concatenated FFT of 8 channels
def returnWindows(data, window, ds, ds_y, label):
    for i in np.arange(0, len(data[1])//window):
        ch_windowed = []
        for ch in data[7:9]:
            ch_windowed.append(ch[i*window:(i*window)+window])
        ds_y.append(label)
        ds.append(np.concatenate((get_fft(ch_windowed))))
    return ds, ds_y # returns 8 channel data, windowed

def gen_file_list(x):
    freq = ['10', '15', '20', '25']
    file_names = []
    for f in freq:
        for i in range(x):
            file_names.append(['eeg-toma-ssvep'+f+'Hz-'+str(i)+'.csv', f])
    return file_names

def test_file_list():
    freq = ['10', '15', '20', '25']
    file_names = []
    for f in freq:
        file_names.append(['eeg-toma-ssvep'+f+'Hz-'+str(4)+'.csv', f])
    return file_names

def main():
    fs = 256 # should get from brainflow board = board_id '0'
    batch_window = 2 * fs
    file_names_freqs = gen_file_list(4)
    test_file_freq = test_file_list()

    test = []
    test_y = []
    train = []
    train_y = []

    # divert files to either test or train --> same recording cannot be split into train/test
    # for n, f in file_names_freqs:
    #     if random.choice([0, 1, 2, 3, 4])==0:
    #         test, test_y = returnWindows(load_data(n), batch_window, test, test_y, f)
    #     else:
    #         train, train_y = returnWindows(load_data(n), batch_window, train, train_y, f)
    # t_files = ['eeg-toma-ssvep05Hz-4.csv', 'eeg-toma-ssvep10Hz-4.csv', 'eeg-toma-ssvep15Hz-4.csv','eeg-toma-ssvep20Hz-4.csv','eeg-toma-ssvep25Hz-4.csv']
    for n, f in file_names_freqs:
        train, train_y = returnWindows(load_data(n), batch_window, train, train_y, f)
    for n, f in test_file_freq:
        test, test_y = returnWindows(load_data(n), batch_window, test, test_y, f)

    print('train: ' + str(len(train)))
    print(collections.Counter(train_y.copy()))
    print('test: ' + str(len(test)))   
    print(collections.Counter(test_y.copy()))

    # start training
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train, train_y)

    out = clf.predict(test)

    print('preditions')
    print(out)
    print('correct')
    print(train_y)

    print('accuracy: '+ str(sklearn.metrics.accuracy_score(test_y, out)))

    sklearn.metrics.plot_confusion_matrix(clf, test, test_y)  
    plt.show()

if __name__ == "__main__":
    main()