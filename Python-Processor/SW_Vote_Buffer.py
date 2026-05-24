from collections import deque
from pylsl import StreamInfo, StreamOutlet
import numpy as np

class VoteBuffer:
    def __init__(self, freq_list, window_time_SI=1.0, step_time_SI=0.04, vote_threshold=0.5):
        self.freq_list = freq_list
        self.window_time = window_time_SI
        self.step_time = step_time_SI
        self.vote_threshold = vote_threshold
        self.vote_window_len = int(self.window_time / self.step_time)
        self.vote_window = deque(maxlen=self.vote_window_len)

        # persistent values for debugging/inspection
        self.winning_freq = None
        self.vote_counts = None

    def connect_to_stream(self, stream_name='SSVEP_Commands', stream_type='Markers'):
        try:
            info = StreamInfo(
                name=stream_name,
                type=stream_type,
                channel_count=1,
                nominal_srate=0,
                channel_format='int32',
                source_id='ssvep_vote_buffer'
            )
            self.LSL_outlet = StreamOutlet(info)
            print(f"LSL outlet '{stream_name}' created successfully.")
        except Exception as e:
            print(f"Unexpected error during connection: {e}")
            raise

    def pull_command(self, cca_index):
        """
        Receives CCA index every 40ms, appends to vote window.
        """
        self.vote_window.append(cca_index)

    def vote(self):
        """
        Tally votes in current window, store result in self.winning_freq.
        """
        if len(self.vote_window) < self.vote_window_len:
            self.winning_freq = None
            self.vote_counts = None
            return

        self.vote_counts = np.bincount(list(self.vote_window), minlength=len(self.freq_list))
        best = np.argmax(self.vote_counts)

        if self.vote_counts[best] / self.vote_window_len > self.vote_threshold:
            self.winning_freq = self.freq_list[best]
        else:
            self.winning_freq = None

    def push_command(self):
        """
        Push self.winning_freq to LSL outlet for Unity to receive.
        """
        if self.winning_freq is not None:
            self.LSL_outlet.push_sample([self.winning_freq])
            print(f"Command sent: {self.winning_freq} Hz")
        else:
            print("Idle — no command sent")