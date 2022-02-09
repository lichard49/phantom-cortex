# File functions
from graph import display_data
from hardware_interfacing import HeadSet, File
from classifying import spectral_analysis
from processing import apply_bandpass, get_fft
from label import import_json

import json
import time

from brainflow import BoardShim


def main():
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

    input = HeadSet("/dev/cu.usbserial-DM0258JS", "testing")

    print(input.params.serial_port)

    input.init_board()
    input.start_session()

    display_data(input, "fft")

    # testing real time classification
    time.sleep(4)
    while (True):
        data = input.board.get_current_board_data(4 * BoardShim.get_sampling_rate(input.board_id))
        apply_bandpass(data, BoardShim.get_sampling_rate(input.board_id), range=(8, 32))
        fft = get_fft(data, BoardShim.get_sampling_rate(input.board_id))
        prediction = spectral_analysis(fft[7])
        time.sleep(0.5)
        print('highest frequency amplitutde is at:' + str(prediction) + ' Hz')

    # testing label.py and json
    # d = import_json("test.json")
    # print(d)

if __name__ == "__main__":
    main()