from brainflow.data_filter import DataFilter, FilterTypes, WindowFunctions, DetrendOperations

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

# performs fast fourier transform on each channel
# param: timeseries (multichannel, idx0-7:ch1-8), sampling rate of recording,
#        minimum frequency of fft (default=0), maximum frequency of fft (default=61)
# output: fft of each channel
def get_fft(timeseries_data, sampling_rate, fft_min=0, fft_max=61):
    fft = [None] * len(timeseries_data)
    for ch, ch_data in enumerate(timeseries_data):
        nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
        psd = DataFilter.get_psd_welch(ch_data, nfft, nfft // 2, sampling_rate,
                                WindowFunctions.BLACKMAN_HARRIS.value)
        fft[ch] = psd[0][fft_min:fft_max]
    return fft