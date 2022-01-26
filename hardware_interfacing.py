# This file contains functionaluty which allows the user to stream and record data from various input sources (i.e. a Headset or File)

# General imports
from asyncore import file_dispatcher
import pandas as pd
import numpy as np
# Brainflow
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
# File functions
from graph import Graph_FFT, Graph_Timeseries

class HeadSet:
    def __init__(self, serial_port: str, filename="default"):
        # not private to the class, should they be?
        # note file name is not necessary, if we are "streaming" from a headset
        self.serial_port = serial_port
        self.filename = filename
        self.board_id = 0
        self.other_info = ''

class File:
    def __init__(self, filename):
        # filename is always required here
        self.serial_port = ''
        self.filename = filename
        self.board_id = -3
        self.other_info = "0"


# Abstract class / parent class
class InputSource(HeadSet, File):
    pass

# Description -- assumption that the board the user is using is the CYTON BOARD

# Notes
# board_id = 0, serial_port = 'COM3' or '/dev/cu.usbserial-#####' for recording / streaming directly from headset 
# board_id = -3, other_info="0" # for cyton and has to be string, filename = "__"
# streaming pre-recorded data

# param: graph_timeseries is a boolean (true/false) that indicates whether the user expects to see a graph of timeseries or 
# output:
def display_data(input_source: InputSource, graph_timseries):

    # activate board and place default values
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    params.ip_port = 0
    params.serial_port = ''
    params.mac_address = ''
    params.other_info = ''
    params.serial_number = ''
    params.ip_address = ''
    params.ip_protocol = 0
    params.timeout = 0
    params.file = ''

    # If data is from a File, update necessary params
    if isinstance(input_source, File):
        params.other_info = "0"

    if isinstance(input_source, HeadSet):
        params.serial_port = input_source.serial_port

    # If data is from  HeadSet and they dont use a file name,
    # then we are just streaming, else we record and if 
    # the data is from a file they must specify a filename
    # so we will also record

    # fix this
    recording = input_source.filename != "default"

    # right now I think we only have the functionality to record time series graphs --> we should change this
    if recording and not graph_timseries:
        print("Cannot record FFT data, functionality not yet implemented.")
        return

    board = BoardShim(input_source.board_id, params)
    board.prepare_session()

    board.start_stream(45000, '')
    
    # If graph_timeseries --> Graph_Timeseries, else --> Graph_FFT
    if graph_timseries:
        Graph_Timeseries(board)
    else:
        Graph_FFT(board)

    if recording:
        data = board.get_board_data()  
        board.stop_stream()
        board.release_session()

        df = pd.DataFrame(np.transpose(data))
        print('Data From the Board')
        print(df.head(10))

        DataFilter.write_file(data, input_source.filename, 'w')  # use 'a' for append mode
        restored_data = DataFilter.read_file(input_source.filename)
        restored_df = pd.DataFrame(np.transpose(restored_data))


def main():
    # Split this into another file
    # This is the program the user will interact with, they should not have to search into other files

    # Tutorial

    # Goal: Graph FFT or timeseries data 
    
    """
    Choose input source: either headset or file

    If you are planning to record: Headset

        input = HeadSet(port_value, filename where data will be saved)

    If you are planning to stream: Headset or File

        input = HeadSet(port_value) -- no filename given
        input = File(filename where we will stream data from)

    Now actually display data:

        display_data(input, true if you want timeseries graph or false if you want FFT graph)

    """

    input = HeadSet("COM3", "testing")
    display_data(input, True)


if __name__ == "__main__":
    main()




