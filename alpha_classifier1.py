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

def ml_model():
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
    return clf

def ml_detect_alpha(data, model):
    print("detecting...")
    # model = ml_model()

    pred = model.predict(data)

    if pred == 1:
        print('alpha detected')
        # beep
    

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
    
    score, perms, pval = model_selection.permutation_test_score(
    clf, ds, ds_y, scoring="accuracy")
    print('preditions')
    print(clf.predict(X_test))
    print('correct')
    print(y_test)

    estimator = clf.estimators_[5]

    from sklearn.tree import export_graphviz
    # Export as dot file
    export_graphviz(estimator, out_file='alpha_tree.dot', 
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'alpha-tree.png', '-Gdpi=600'])

    # Display in jupyter notebook
    from IPython.display import Image
    Image(filename = 'alpha-tree.png')

if __name__ == "__main__":
    main()


