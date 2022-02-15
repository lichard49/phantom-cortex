# File functions
from graph import display_data, test_display_data
from hardware_interfacing import HeadSet, File
from classifying import spectral_analysis
from processing import get_fft
from label import import_json
# from hardware_interfacing import queue_classification
import globals
# General imports
import json
import time
# Brainflow
from brainflow import BoardShim


def main():
    """
    Choose an input source: either a HeadSet or a File

    If you are planning to record, use a Headset:

        input_source = HeadSet(port_value, filename where data will be saved)

    If you are planning to stream, use a Headset or File:

        input_source = HeadSet(port_value)
            --> no filename given as we are not recording data

        input_source = File(filename where we will stream data from)

    Once an input source has been selected, set up and start gathering data from your board:
        
        input_source.init_board()
        input_source.start_session()

    Now display the data collected:

        display_data(input_source, string that represents graph type you wish displayed)
            --> right now the only graph types available are: "timeseries" or "fft"

    NOTE: Before running this program make sure to initialize all global variables like such: globals.init()
    """
    
    test_experiment_labels = {
        "filename": "test.py",
        "user": "Zage Strassberg-Phillips",
        "headset": "EEG",
        "board": "Cyton",
        "channels": "8"
    }

    globals.init()
    test_display_data(test_experiment_labels)
    globals.printClassifyQueue()

if __name__ == "__main__":
    main()