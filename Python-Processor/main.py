'''
This is where we run the Processor (filter + CCA + SWVD)
'''
from collections import deque
import numpy as np
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet

from SW_Data_Buffer import DataBuffer
from SW_Vote_Buffer import VoteBuffer
from CCA import perform_CCA, build_reference_signals

# ### ENCODING/"MACROS" ###

#Frequencies (in Hz)
UP = 7
LEFT = 11    
DOWN = 13
RIGHT = 17

def main():
    # set up
    data_buffer = DataBuffer(n_channels=8,sampling_rate_hz=250)
    vote_buffer = VoteBuffer(freq_list=(UP,LEFT,DOWN,RIGHT))

    # reference signals
    ref_signals = build_reference_signals(freq_list=(UP,LEFT,DOWN,RIGHT), n_samples=250, sampling_rate_hz=250)


    # LSL
    data_buffer.connect_to_stream()
    vote_buffer.connect_to_stream()

    #loop
    while True:
        data_buffer.pull_step(step_time_secs=0.04) # 40ms
        cca_freq = perform_CCA(data_buffer=data_buffer, reference_signals=ref_signals, freq_list=(UP,LEFT,DOWN,RIGHT))
        vote_buffer.pull_command(freq=cca_freq)
        vote_buffer.vote()
        vote_buffer.push_command()

if __name__ == "__main__": main()