import argparse
import time
import logging
import random
import numpy as np
import pandas as pd

# python3 stream-data.py --board-id=-3 --other-info="0" --file="eeg-1.csv"

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyautogui # for spacebar

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

import scipy
from scipy import signal
# import utils

# read and return np array of data (default), pd df if form='pd'
def load_data(self, filename, form='np'):
    readdata = DataFilter.read_file(filename)
    if form == 'pd':
        data_df = pd.DataFrame(np.transpose(readdata))
        return data_df
    return readdata

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 10
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot',size=(800, 600))

        # init curves and plots --> basically the same code, comment one out
        # self._init_timeseries()
        self._init_fft()

        timer = QtCore.QTimer()
        # timer.timeout.connect(self.update)
        timer.timeout.connect(self.updateFFT)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        self.ranges = list()        # tuples of min / max
        self.allrange = [-150, 150] # [-2**12, 2**12] is hardware's limit
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left')
            p.setMenuEnabled('left', False)
            p.showAxis('bottom')
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def _init_fft(self):
        self.plots = list()
        self.curves = list()
        # self.allrange = [0, 20]
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left')
            p.setMenuEnabled('left', False)
            p.showAxis('bottom')
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('FFT Plot')
            # p.setYRange(self.allrange[0], self.allrange[1])
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
    
    # update for 1 Hz binned FFT
    def updateFFT(self):
        data = self.board_shim.get_current_board_data(self.num_points) # 8 ecg channels, each num pts long
        avg_bands = [0, 0, 0, 0, 0]
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 51.0, 100.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 51.0, 100.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 50.0, 4.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 4.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)
            psd = DataFilter.get_psd_welch(data[channel], nfft, nfft // 2, self.sampling_rate,
                                   WindowFunctions.BLACKMAN_HARRIS.value)
            # self.plots[count].setYRange(self.fftrange[0], self.fftrange[1])

            self.curves[count].setData(psd[0][:61]) # cut frequencies at 60Hz
            
        self.app.processEvents()

    # update for timeseries
    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        avg_bands = [0, 0, 0, 0, 0]
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 51.0, 100.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 51.0, 100.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 50.0, 4.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 60.0, 4.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.plots[count].setYRange(self.allrange[0], self.allrange[1])

            self.curves[count].setData(data[channel].tolist())
            
        self.app.processEvents()

def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.CYTON_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    # start 
    board = BoardShim(args.board_id, params)
    board.prepare_session()
    board.start_stream(45000, args.streamer_params)
    
    fs = board.get_sampling_rate(0)

    data = board.get_board_data(1 * fs)
    filt = [64.61473451327436,
    110.51373451327433,
    -47.20126548672565,
    -118.49026548672566,
    10.415734513274344,
    110.02573451327436,
    -7.6502654867256545,
    -111.16626548672565,
    -44.76026548672566,
    86.58773451327434,
    64.12673451327436,
    -93.09926548672564,
    -75.52126548672567,
    87.56473451327435,
    106.11873451327435,
    -73.08026548672566,
    -109.70126548672565,
    49.47873451327434,
    93.91173451327435,
    -30.599265486725656,
    -97.00626548672565,
    7.9747345132743455,
    92.44673451327435,
    -8.138265486725654,
    -103.84226548672567,
    -36.459265486725656,
    118.32573451327434,
    69.00973451327434,
    -84.79926548672566,
    -49.154265486725656,
    64.12673451327436,
    67.54473451327434,
    -97.49426548672565,
    -153.15826548672567,
    22.62273451327435,
    96.84173451327436,
    -55.990265486725654,
    -164.87726548672566,
    -37.435265486725655,
    74.86873451327435,
    -38.41226548672565,
    -198.56826548672566,
    -163.41226548672566,
    -20.834265486725656,
    -61.36126548672566,
    -230.30626548672564,
    -259.60326548672566,
    -127.27926548672565,
    -80.40426548672565,
    -228.35326548672566,
    -289.87726548672566,
    -156.08826548672565,
    -60.87326548672566,
    -157.06426548672565,
    -256.18526548672565,
    -180.01326548672566,
    -80.89226548672565,
    -149.25226548672566,
    -272.2992654867257,
    -221.02926548672568,
    -61.84926548672565,
    -90.17026548672567,
    -237.14226548672568,
    -228.35326548672566,
    -61.84926548672565,
    -21.810265486725655,
    -188.31426548672567,
    -260.09226548672564,
    -100.91226548672566,
    20.181734513274343,
    -122.88526548672564,
    -252.27926548672568,
    -135.09226548672567,
    8.950734513274345,
    -42.80626548672566,
    -193.19726548672566,
    -137.53326548672567,
    32.87673451327434,
    3.579734513274346,
    -154.13526548672567,
    -189.29126548672565,
    -29.135265486725654,
    90.00573451327435,
    -40.85326548672566,
    -132.65026548672566,
    4.068734513274347,
    103.18973451327435,
    20.181734513274343,
    -74.54526548672567,
    0.16173451327434663,
    145.18173451327434,
    113.44373451327434,
    -31.576265486725653,
    16.763734513274347,
    158.85373451327436,
    151.52973451327435,
    27.993734513274347,
    -3.744265486725654,
    146.64673451327434,
    197.42773451327434,
    67.05673451327434,
    8.950734513274345,
    105.14273451327435,
    234.53673451327435,
    145.18173451327434,
    -10.092265486725655,
    51.91973451327434,
    206.21673451327433,
    183.26773451327432,
    12.857734513274348,
    14.810734513274344,
    161.78273451327433,
    196.4507345132743,
    59.24373451327434,
    19.693734513274347,
    141.76373451327433,
    195.47473451327434,
    101.72473451327434,
    -26.693265486725654,
    49.966734513274346,
    196.93973451327435,
    123.20873451327432,
    -1.7912654867256546,
    21.646734513274343,
    157.38873451327436,
    189.12673451327436,
    8.950734513274345,
    -15.463265486725653,
    130.53273451327433,
    176.91973451327434,
    80.72873451327433,
    5.044734513274346,
    106.60773451327435,
    189.61473451327433,
    80.23973451327436,
    -41.83026548672566,
    60.708734513274344,
    165.2007345132743,
    74.38073451327435,
    -69.17426548672566,
    -34.99426548672565,
    129.06873451327434,
    106.60773451327435,
    -68.19726548672566,
    -41.34226548672565,
    126.13873451327433,
    148.59973451327434,
    6.021734513274346,
    -71.61526548672566,
    37.759734513274346,
    128.09173451327433,
    37.759734513274346,
    -98.95926548672566,
    -15.463265486725653,
    154.45873451327432,
    111.97873451327433,
    -50.13126548672566,
    -54.52526548672565,
    100.25973451327434,
    109.04873451327435,
    -41.34226548672565,
    -81.38126548672565,
    51.43173451327434,
    115.88473451327434,
    -1.3032654867256541,
    -96.51726548672565,
    -9.603265486725654,
    109.53673451327435,
    42.15473451327435,
    -91.14626548672567,
    -21.810265486725655,
    117.83773451327434,
    81.21673451327436,
    -76.01026548672564,
    -83.33426548672566,
    88.54073451327434,
    108.56073451327435,
    -42.31826548672566,
    -68.19726548672566,
    64.61473451327436,
    134.92773451327434,
    8.462734513274345,
    -99.44726548672566,
    -0.8142654867256542,
    134.92773451327434,
    64.61473451327436,
    -86.75226548672566,
    -41.34226548672565,
    95.86473451327436,
    88.05273451327434,
    -67.70926548672566,
    -96.02926548672565,
    83.16973451327434,
    139.81073451327435,
    -3.744265486725654,
    -87.72826548672566,
    36.29473451327434,
    134.43973451327435,
    13.833734513274347,
    -102.37726548672566,
    -10.092265486725655,
    134.43973451327435,
    73.89273451327435,
    -69.66226548672566,
    -46.224265486725656,
    90.98273451327435,
    97.32973451327433,
    -46.71326548672565,
    -77.96326548672565,
    52.40773451327435,
    111.00173451327436,
    -31.576265486725653,
    -121.90826548672564,
    14.322734513274348,
    111.00173451327436,
    6.021734513274346,
    -114.09526548672565,
    -34.01726548672565,
    124.18573451327435,
    49.47873451327434,
    -108.23626548672567,
    -59.89626548672566,
    79.26373451327433,
    92.44673451327435,
    -69.66226548672566,
    -97.00626548672565]

    EPOCH_LENGTH = 0.2
    BUFFER_LENGTH = 5
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))

    print('starting')
    while True:
        data = board.get_current_board_data(int(EPOCH_LENGTH * fs))
        if len(data[1]) >= int(EPOCH_LENGTH * fs):

            DataFilter.detrend(data[1], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[1], fs, 51.0, 100.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandpass(data[1], fs, 51.0, 100.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[1], fs, 50.0, 4.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[1], fs, 60.0, 4.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)

            data_epoch = data[1]
            print(data_epoch.shape)
            matchFilt = signal.hilbert(filt)

            matches = signal.correlate(matchFilt,data_epoch)

            matchesAbs = np.abs(matches[:])

            maxMatch = np.max(matchesAbs)/1e5
            print(maxMatch)

            if maxMatch > 95:
                print('blink')  
                pyautogui.press("space")



if __name__ == "__main__":
    main()           