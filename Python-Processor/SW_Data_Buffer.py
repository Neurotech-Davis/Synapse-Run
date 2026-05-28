from collections import deque
import numpy as np
from pylsl import StreamInlet, resolve_streams
import logging #for buffered print

class DataBuffer:
    def __init__(self, n_channels, sampling_rate_hz, window_time_secs = 1.0,):
        self.num_chs = n_channels
        self.freq = sampling_rate_hz
        self.window_time = window_time_secs

        # compute window length in terms of NUMBER OF SAMPLES
        self.window_len = self.freq * self.window_time

        # create an internal deque that implements the sliding window structure
        self.sliding_window = deque(maxlen = self.window_len)
    
    def ret_sliding_window(self):
        return self.sliding_window
    
    def connect_to_stream(self, type = 'EEG', timeout = 10.0):
        try:
            self.streams = resolve_streams('type', type, timeout)
            if not self.streams:
                raise TimeoutError(f"No {type} found.")
            self.LSL_inlet = StreamInlet(self.streams[0])

            # validation of inlet specs with arguments passed to innit
            specs = self.LSL_inlet.info()
            actual_freq = specs.nominal_srate()
            actual_num_chs = specs.channel_count()
            if self.freq != actual_freq:
                raise ValueError(
                    f"Expected sampling rate {self.freq}Hz "
                    f"but stream is broadcasting at {actual_freq}Hz"
            )
            if self.num_chs != actual_num_chs:
                raise ValueError(
                    f"Expected {self.num_chs} channels "
                    f"but stream has {actual_num_chs}"
            )

        except TimeoutError as e:
            print(f"Connection timed out: {e}")
            raise
        except ValueError as e:
            print(f"Hardware mismatch: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during connection: {e}")
            raise
    
    def pull_step(self, step_time_secs = 0.01):
        try:
            self.step_time = step_time_secs
            # compute step length in terms of NUMBER OF SAMPLES
            self.step_len = self.freq * self.step_time
            number_of_samples_to_collect = self.step_len
            while number_of_samples_to_collect:
                samples, _ = self.LSL_inlet.pull_chunk(max_samples=number_of_samples_to_collect)
                print(f"pulled {samples} from LSL")
                print("Adding to buffer")
                self.sliding_window.extend(samples)
                if len(self.sliding_window) == self.sliding_window.maxlen:
                    print(f"leftmost {len(samples)} samples ejected to add {len(samples)} samples")
                number_of_samples_to_collect -= len(samples)
        except:
            pass

 