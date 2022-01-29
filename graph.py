# This file contains two graph classes, Graph_Timeseries and Graph_FFT, which graph timeseries and fast fourier transform data 

# General imports
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
# Brainflow 
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
# File functions
from processing import standard_filter_timeseries
from hardware_interfacing import InputSource, HeadSet, record

# Graph_Timeseries Class
class Graph_Timeseries:
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

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
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
    
    # update timeseries data
    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        # filtering
        standard_filter_timeseries(data, self.sampling_rate)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            self.plots[count].setYRange(self.allrange[0], self.allrange[1])
            self.curves[count].setData(data[channel].tolist())
        self.app.processEvents()

# Graph_FFT Class
class Graph_FFT:
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

        self._init_fft()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.updateFFT)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

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
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
    
    # update for 1 Hz binned FFT
    def updateFFT(self):
        data = self.board_shim.get_current_board_data(self.num_points) # 8 ecg channels, each num pts long
         # filtering
        standard_filter_timeseries(self.exg_channels, self.sampling_rate)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)
            psd = DataFilter.get_psd_welch(data[channel], nfft, nfft // 2, self.sampling_rate,
                                   WindowFunctions.BLACKMAN_HARRIS.value)
            self.curves[count].setData(psd[0][:61]) # cut frequencies at 60Hz 
        self.app.processEvents()

# Description -- assumption that the board the user is using is the CYTON BOARD

# Notes
# board_id = 0, serial_port = 'COM3' or '/dev/cu.usbserial-#####' for recording / streaming directly from headset 
# board_id = -3, other_info="0" # for cyton and has to be string, filename = "__"
# streaming pre-recorded data

# param: graph_timeseries is a string that indicates whether the user expects to see a graph of timeseries or 
# output:
def display_data(input_source: InputSource, graph_timeseries):

    # activate board
    BoardShim.enable_dev_board_logger()

    # check if recording
    recording = isinstance(input_source, HeadSet) and input_source.filename != None

    if recording and graph_timeseries == "fft":
        print("Cannot record FFT data.")
        return

    board = BoardShim(input_source.board_id, input_source.params)
    board.prepare_session()

    board.start_stream(45000, '')
    
    # graphing
    if graph_timeseries == "timeseries":
        Graph_Timeseries(board)
    elif graph_timeseries == "fft":
        Graph_FFT(board)

    if recording:
        record(board, input_source.filename)
