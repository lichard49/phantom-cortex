# General imports
from asyncore import file_dispatcher
from cmath import exp
from tkinter import E
from xml.dom.expatbuilder import parseString
from xml.dom.minidom import Element
import pandas as pd
import numpy as np
from queue import Queue
from label import write_json
# Brainflow
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
# File variables
import globals

# Abstract class / parent class
class InputSource:
    def __init__(self):
        self.params = BrainFlowInputParams()
        self.params.ip_port = 0
        self.params.serial_port = ''
        self.params.mac_address = ''
        self.params.other_info = ''
        self.params.serial_number = ''
        self.params.ip_address = ''
        self.params.ip_protocol = 0
        self.params.timeout = 0
        self.params.file = ''
        # users board
        self.board = None

    """
    Initializes the users CYTON BOARD
    """
    def init_board(self):
        self.board = BoardShim(self.board_id, self.params)

    """
    Starts streaming from the users board
    """
    def start_session(self):
        if self.board is None:
            self.init_board()
        self.board.prepare_session()
        self.board.start_stream(45000, '')

    """
    Stop streaming from the users board
    """
    def stop_session(self):
        self.board.stop_stream()

# HeadSet input source (recording and streaming capability)
class HeadSet(InputSource):
    def __init__(self, serial_port: str, filename = None):
        InputSource.__init__(self)
        self.params.serial_port = serial_port
        self.filename = filename
        self.board_id = 0

# File input source (streaming capability)
class File(InputSource):
    def __init__(self, filename):
        InputSource.__init__(self)
        self.filename = filename
        self.board_id = -3
        self.params.other_info = "0"

"""
Records timeseries data from board and stores it in the given filename

params: the users current board where they are gathering data,
        a filename where the data will be stored
output: nothing is explicitly returned
"""
def record(board):
    # while we are collecting data pull from the board
    while globals.collecting_data:
        data = board.get_board_data() # all data from a board, removes from ringbuffer
        globals.queue_classification.put(data)
        globals.queue_file_data_transfer.put(data)

# TESTING
def test_record(board: Queue):
    # while we are collecting data pull from the board
    while globals.collecting_data:
        if not board.empty() and not board.closed:
            data = board.get()
            globals.queue_classification.put(data)
            globals.queue_file_data_transfer.put(data)

"""
Stores recording data into a designated file specified by the user

params: a string representing a filename
output: none
"""
def file_data_transfer(filename):
    # while we are collecting data pull from the queue
    while globals.collecting_data:
        if not globals.queue_file_data_transfer.empty():
            data = globals.queue_file_data_transfer.get()
            DataFilter.write_file(data, filename, 'a')

    # once we are no longer collecting data, we need to empty the queue
    while not globals.queue_file_data_transfer.empty():
        data = globals.queue_file_data_transfer.get()
        DataFilter.write_file(data, filename, 'a')

# TESTING
def test_file_data_transfer(filename, experiment_labels):
    f = open(filename, "w")

    # while we are collecting data pull from the queue
    while globals.collecting_data:
        if not globals.queue_file_data_transfer.empty():
            data = globals.queue_file_data_transfer.get()
            # for testing
            f.write(str(data))

    # once we are no longer collecting data, we need to empty the queue
    while not globals.queue_file_data_transfer.empty():
        data = globals.queue_file_data_transfer.get()
        # for testing
        f.write(str(data))
    
    # for testing
    f.close()

    # NOW ADD FILE TO JSON
    write_json(experiment_labels)