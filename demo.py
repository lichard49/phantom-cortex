# File functions
from graph import display_data
from hardware_interfacing import HeadSet, File

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
    display_data(input, "timeseries")


if __name__ == "__main__":
    main()