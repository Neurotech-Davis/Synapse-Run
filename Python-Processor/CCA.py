import numpy as np
def build_reference_signals(freq_list, n_samples, sampling_rate, n_harmonics=2):
    # if works sus, try n_harmonics=3
    """
    the whole matrix X:
    (9, 250) -> 9 electrodes, 250 samples (1-second time period)
              t=1/250  t=2/250  t=3/250  ...  t=250/250
    Oz       [ 0.23,   0.21,    0.19,   ...,  0.31    ]   ← electrode 1
    O1       [ 0.11,   0.13,    0.10,   ...,  0.09    ]   ← electrode 2
    O2       [ 0.19,   0.17,    0.21,   ...,  0.14    ]   ← electrode 3
    P3       [ 0.05,   0.04,    0.06,   ...,  0.03    ]   ← electrode 4
    P4       [ 0.07,   0.08,    0.05,   ...,  0.06    ]   ← electrode 5
    Pz       [ 0.12,   0.11,    0.13,   ...,  0.10    ]   ← electrode 6
    P7       [ 0.03,   0.04,    0.02,   ...,  0.05    ]   ← electrode 7
    P8       [ 0.06,   0.05,    0.07,   ...,  0.04    ]   ← electrode 8
    CPz      [ 0.08,   0.09,    0.07,   ...,  0.11    ]   ← electrode 9
    """
    t = np.arange(1, n_samples + 1) / sampling_rate
    references = []
    
    for freq in freq_list:
        Y = []
        for h in range(1, n_harmonics + 1):
            Y.append(np.sin(2 * np.pi * h * freq * t))
            Y.append(np.cos(2 * np.pi * h * freq * t))
        references.append(np.array(Y))
    
    return references  # list of arrays, each shape (2*n_harmonics, n_samples)

def perform_CCA(data_buffer, reference_signals):
    return None