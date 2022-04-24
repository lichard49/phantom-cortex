# File functions
from graph import display_data
from hardware_interfacing import HeadSet, File
from classifying import spectral_analysis
from processing import apply_bandpass, get_fft
from label import import_json
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

    NOTE: You cannot display data and attempt to use real-time classification at the same time
    """

    input_source = HeadSet("/dev/cu.usbserial-DM0258JS", "testing")

    # testing real time classification
    time.sleep(4)
    # while (True):
    #     data = input.board.get_current_board_data(4 * BoardShim.get_sampling_rate(input.board_id))
    #     apply_bandpass(data, BoardShim.get_sampling_rate(input.board_id), range=(8, 32))
    #     fft = get_fft(data, BoardShim.get_sampling_rate(input.board_id))
    #     prediction = spectral_analysis(fft[7])
    #     time.sleep(0.5)
    #     print('highest frequency amplitutde is at:' + str(prediction) + ' Hz')
    input_source.init_board()
    input_source.start_session()

    # displaying data
    display_data(input_source, "fft")

    # real time classification
    # time.sleep(4)
    # while (True):
    #     data = input_source.board.get_current_board_data(4 * BoardShim.get_sampling_rate(input_source.board_id))
    #     fft = get_fft(data, BoardShim.get_sampling_rate(input_source.board_id))
    #     prediction = spectral_analysis(fft[7], [(10, 14), (15, 19), (20, 24), (25, 29)], [12, 17, 22, 27])
    #     time.sleep(0.5)
    #     print(prediction)

if __name__ == "__main__":
    main()