from collections import deque
import numpy as np
from pylsl import StreamInlet, resolve_stream

class DataBuffer:
    def __init__(self, num_channels, device_freq_hz, step_time_SI = 0.001, window_time_SI = 1.0,):
        self.num_chs = num_channels
        self.freq = device_freq_hz
        self.step_time = step_time_SI
        self.window_time = window_time_SI

        # compute window and step length in terms of NUMBER OF SAMPLES
        self.window_len = self.freq * self.window_time
        self.step_len = self.freq * self.step_time

        # create an internal deque that implements the sliding window structure
        self.sliding_window = deque(maxlen = self.window_len)
    
    def connect_to_stream(self, type = 'EEG', timeout = 10.0):
        try:
            self.streams = resolve_stream('type', type, timeout)
            if not self.streams:
                raise TimeoutError("No {type} found.")
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


 