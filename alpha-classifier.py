import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
import numpy as np
import pandas as pd
from brainflow import DataFilter, board_shim, WindowFunctions
import matplotlib.pyplot as plt


def load_data(filename, form='np'):
    readdata = DataFilter.read_file(filename)
    if form == 'pd':
        data_df = pd.DataFrame(np.transpose(readdata))
        return data_df
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

# take in data 23 channels after trimming 
def returnWindows(data, window, ds, ds_y, label):
    for i in np.arange(0, len(data[1])//window):
        ch_windowed = []
        for ch in data[1:9]:
            ch_windowed.append(ch[i*window:(i*window)+window])
        ds_y.append(label)
        ds.append(np.concatenate((get_fft(ch_windowed))))
    return ds, ds_y # returns 8 channel data, windowed

def main():
    fs = 256 # should get from brainflow board = board_id '0'

    data1 = load_data('eeg-alpha-1.csv')
    data2 = load_data('eeg-alpha-2.csv')
    data3 = load_data('eeg-alpha-3-closed.csv')
    datab1 = load_data('eeg-ssvep-20hz.csv')
    datab2 = load_data('eeg-ssvep-15hz.csv')
    
    batch_window = 2 * fs

    # think of a way to window + label more robustly
    ds = []
    ds_y = []
    ds, ds_y = returnWindows(data1, batch_window, ds, ds_y, 1)
    ds, ds_y = returnWindows(data2, batch_window, ds, ds_y, 1)
    ds, ds_y = returnWindows(data3, batch_window, ds, ds_y, 1)
    ds, ds_y = returnWindows(datab1, batch_window, ds, ds_y, 0)
    ds, ds_y = returnWindows(datab2, batch_window, ds, ds_y, 0)
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(ds, ds_y, test_size=0.33)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    
    # score, perms, pval = model_selection.permutation_test_score(
    # clf, ds, ds_y, scoring="accuracy")
    print('preditions')
    print(clf.predict(X_test))
    print('correct')
    print(y_test)

    # print(score)
    # print(perms)
    # print(pval)

    # fig, ax = plt.subplots()

    # ax.hist(perms, bins=20, density=True)
    # ax.axvline(score, ls="--", color="r")
    # score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {pval:.3f})"
    # ax.text(0.7, 10, score_label, fontsize=12)
    # ax.set_xlabel("Accuracy score")
    # _ = ax.set_ylabel("Probability")
    # plt.show()

if __name__ == "__main__":
    main()


