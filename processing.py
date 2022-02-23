# Brainflow
from brainflow.data_filter import DataFilter, FilterTypes, WindowFunctions, DetrendOperations

"""
Applies "standard" filters to raw timeseries (as deemed by BrainFlow)

params: raw timeseries (multichannel, idx0-7:ch1-8)
output: filtered timeseries (same dimensions)
"""
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

# applies bandpass for given range to "amplify" frequencies we care about 
# params: multi-channel timeseries data, sampling rate, and optional range
# output: applies bandpass to timeseries data (same dimensions)
def apply_bandpass(timeseries_data, sampling_rate, order=2, range=(10, 40)):
    for ch, ch_data in enumerate(timeseries_data):
        DataFilter.perform_bandpass(ch_data, sampling_rate, range[0], range[1], order,
                                    FilterTypes.BUTTERWORTH.value, 0)

"""
Performs fast fourier transform on each channel

params: timeseries (multichannel, idx0-7:ch1-8), sampling rate of recording,
        minimum frequency of fft (default=0), maximum frequency of fft (default=61)
output: fft of each channel
"""
def get_fft(timeseries_data, sampling_rate, fft_min=0, fft_max=61):
    fft = [None] * len(timeseries_data)
    for ch, ch_data in enumerate(timeseries_data):
        nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
        psd = DataFilter.get_psd_welch(ch_data, nfft, nfft // 2, sampling_rate,
                                WindowFunctions.BLACKMAN_HARRIS.value)
        fft[ch] = psd[0][fft_min:fft_max]
    return fft

def get_fft_ch(timeseries_data, sampling_rate, fft_min=0, fft_max=61):
    fft = [None] * len(timeseries_data)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    psd = DataFilter.get_psd_welch(timeseries_data, nfft, nfft // 2, sampling_rate,
                            WindowFunctions.BLACKMAN_HARRIS.value)
    fft = psd[0][fft_min:fft_max]
    return fft