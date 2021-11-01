import argparse
import time
import logging
import random
import numpy as np
import pandas as pd

# python3 stream-data.py --board-id=-3 --other-info="0" --file="eeg-1.csv"

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

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

        self._init_timeseries()
        self._init_fft()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.updateFFT)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        self.ranges = list() # tuples of min / max
        self.allrange = [-150, 150] # [-2**12, 2**12]
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left')
            p.setMenuEnabled('left', False)
            p.showAxis('bottom')
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            # self.ranges.append([-2**12, 2**12])
            
            curve = p.plot()
            self.curves.append(curve)

    def _init_fft(self):
        self.plots = list()
        self.curves = list()

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

            self.curves[count].setData(psd[0])
            
        self.app.processEvents()


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
            # update ylim based on min, max values --> experiencing latency? issues in update
            # if len(data[channel]) >= self.num_points:
            #     if min(data[channel]) < self.ranges[count][0]:
            #         self.ranges[count][0] = min(data[channel])
            #     if max(data[channel]) > self.ranges[count][1]:
            #         self.ranges[count][1] = max(data[channel])
            # self.plots[count].setYRange(self.ranges[count][0], self.ranges[count][1])
            # if self.ranges[count][1] > self.allrange[1]:
            #     self.allrange[1] = self.ranges[count][1]
            #     self.plots[count].setYRange(self.allrange[0], self.allrange[1])
            # if self.ranges[count][0] > self.allrange[0]:
            #     self.allrange[0] = self.ranges[count][0]
            #     self.plots[count].setYRange(self.allrange[0], self.allrange[1])
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
    params.other_info = str(BoardIds.CYTON_BOARD.value)
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    # params = BrainFlowInputParams ()
    # params.other_info = str(BoardIds.CYTON_BOARD.value)
    # params.file = 'eeg-1.csv'

    # print(BoardIds.PLAYBACK_FILE_BOARD.value)

    board = BoardShim(args.board_id, params)

    print(board.get_board_data)

    # --------------------------------------
    # board = BoardShim(args.board_id, params)
    # filename = 'file://eeg-1.csv:w'
    board.prepare_session()
    board.start_stream(45000, args.streamer_params)
    g = Graph(board)

if __name__ == "__main__":
    main()