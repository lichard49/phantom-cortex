from time import time
import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations


# applies "standard" filters to raw timeseries (as deemed by BrainFlow)
# param: raw timeseries (multichannel, idx0-7:ch1-8)
# output: filtered timeseries (same dimensions)
def standard_filter_timeseries(timeseries_data, sampling_rate):
    for ch, ch_data in enumerate(timeseries_data):
        DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(ch_data, sampling_rate, 51.0, 100.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandpass(ch_data, sampling_rate, 51.0, 100.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(ch_data, sampling_rate, 50.0, 4.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(ch_data, sampling_rate, 60.0, 4.0, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)

# no loop version
# def standard_filter_timeseries(timeseries_data, sampling_rate):
#     DataFilter.detrend(timeseries_data, DetrendOperations.CONSTANT.value)
#     DataFilter.perform_bandpass(timeseries_data, sampling_rate, 51.0, 100.0, 2,
#                                 FilterTypes.BUTTERWORTH.value, 0)
#     DataFilter.perform_bandpass(timeseries_data, sampling_rate, 51.0, 100.0, 2,
#                                 FilterTypes.BUTTERWORTH.value, 0)
#     DataFilter.perform_bandstop(timeseries_data, sampling_rate, 50.0, 4.0, 2,
#                                 FilterTypes.BUTTERWORTH.value, 0)
#     DataFilter.perform_bandstop(timeseries_data, sampling_rate, 60.0, 4.0, 2,
#                                         FilterTypes.BUTTERWORTH.value, 0)

# performs fast fourier transform on each channel
# param: raw timeseries (multichannel, idx0-7:ch1-8)
# output: filtered timeseries (same dimensions)
def get_fft(timeseries_data, sampling_rate):
    fft = list(len(timeseries_data))
    for ch, ch_data in enumerate(timeseries_data):
        nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
        psd = DataFilter.get_psd_welch(ch_data, nfft, nfft // 2, sampling_rate,
                                WindowFunctions.BLACKMAN_HARRIS.value)
    return fft