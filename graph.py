# General imports
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
# Brainflow 
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
# File functions
from processing import standard_filter_timeseries
from hardware_interfacing import InputSource, HeadSet, record, test_file_data_transfer, test_record, file_data_transfer
import globals
# Threading
import threading
import time
from queue import Queue

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

        # if you are recording, you have to place this data into a record global queue to be transfered
        if globals.collecting_data:
            data = self.board_shim.get_board_data() # all data from a board, removes from ringbuffer
            globals.queue_board_data.put(data)

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
        standard_filter_timeseries(data, self.sampling_rate)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)
            psd = DataFilter.get_psd_welch(data[channel], nfft, nfft // 2, self.sampling_rate,
                                   WindowFunctions.BLACKMAN_HARRIS.value)
            self.curves[count].setData(psd[0][:61]) # cut frequencies at 60Hz 
        self.app.processEvents()

"""
Displays a timeseries or fft graph, and simultanesoulst records timeseries data depending on input
NOTE: This function works under the assumption the board the user is using is the CYTON BOARD

params: an input source (i.e. a HeadSet or File), a string indicating what graph type to display
output: a graph is displayed and data is potentially recorded, but no output is explicity returned
"""
def display_data(input_source: InputSource, graph_type):
    # activate board
    BoardShim.enable_dev_board_logger()

    # we begin collecting data
    globals.startCollectingData()

    # check if recording
    recording = isinstance(input_source, HeadSet) and input_source.filename != None

    # If we are recording, we will start two threads: (1) a recording thread and (2) a file thread
    thread_record = None
    thread_file_data_transfer = None

    if recording: 
        # start the recording stream
        thread_record = threading.Thread(target=record, args=())
        thread_record.start()

        # start the file stream
        thread_file_data_transfer = threading.Thread(target=file_data_transfer, args=(input_source.filename, ))
        thread_file_data_transfer.start()

    # Start graphing thread (i.e. continue running main)
    graph(input_source, graph_type)

    # On the main thread once the graphing is completed, gathering data is complete
    globals.stopCollectingData()

    # Join recording thread
    if thread_record != None: 
        thread_record.join()
        thread_file_data_transfer.join()


"""
Displays a graph based on given graph type

params: an InputSource (HeadSet or File) and a string representin the graph type to be displayed
output: none
"""
def graph(input_source: InputSource, graph_type):
    # graphing
    if graph_type == "timeseries":
        Graph_Timeseries(input_source.board)
    elif graph_type == "fft":
        Graph_FFT(input_source.board)


#TESTING METHODS
def test_display_data(experiment_labels):
    # global collecting_data
    q = Queue()
    q.closed = False

    # we begin collecting data
    globals.startCollectingData()

    # start the recording stream
    thread_record = threading.Thread(target=test_record, args=(q,))
    thread_record.start()

    # start the file stream
    thread_file_data_transfer = threading.Thread(target=test_file_data_transfer, args=("test", experiment_labels,))
    thread_file_data_transfer.start()

    for i in range(100000000000):
        print(i)
        q.put(i)

    time.sleep(20)

    # On the main thread once the graphing is completed, gathering data is complete
    globals.stopCollectingData()

    thread_record.join()
    thread_file_data_transfer.join()