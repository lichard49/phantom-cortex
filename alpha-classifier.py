import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from brainflow import DataFilter, board_shim, WindowFunctions


def load_data(filename, form='np'):
    readdata = DataFilter.read_file(filename)
    if form == 'pd':
        data_df = pd.DataFrame(np.transpose(readdata))
        return data_df
    return readdata

def get_fft(data):
    fs = 256
    fft = []
    for ch in data[1:9]:
        nfft = DataFilter.get_nearest_power_of_two(fs)
        psd = DataFilter.get_psd_welch(ch, nfft, nfft // 2, fs,
                                WindowFunctions.BLACKMAN_HARRIS.value)
        fft.append(psd[0][:61])
    return fft

def main():
    fs = 256 # should get from brainflow board = board_id '0'

    data1 = load_data('eeg-alpha-1.csv')
    data2 = load_data('eeg-alpha-2.csv')
    data3 = load_data('eeg-alpha-3-closed.csv')
    datab1 = load_data('eeg-ssvep-20hz.csv')
    # cut off the edges
    data1 = data1[int(5*fs):len(data1)-int(5*fs)]
    data2 = data2[int(5*fs):len(data2)-int(5*fs)]
    data3 = data3[fs:len(data3)-fs] # one second because of the nature of recording
    
    batch_window = 2 * fs

    fft1 = get_fft(data1)
    fft2 = get_fft(data2)
    fft3 = get_fft(data3)
    fftb1 = get_fft(datab1)

    t = []
    t.append(fft1)
    t.append(fft2)
    t.append(fftb1)

    clf = RandomForestClassifier(random_state=0)
    clf.fit(t, [1, 1, 0])
    
    print(clf.predict([data3]))

if __name__ == "__main__":
    main()


