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

    matrix y in Y:
     (4, 250) -> (2 harmonics * 2 waves per harmonic), 250 samples 
     each harmonic has 2 waves ("2 waves per harmonic"): one sine wave and one cosine wave

                  t=1/250  t=2/250  t=3/250  ...  t=250/250
    sin(1 x 17t) [ 0.41,   0.80,    0.99,   ...,  0.41    ]   ← harmonic 1 sine
    cos(1 x 17t) [ 0.91,   0.60,   -0.14,   ...,  0.91    ]   ← harmonic 1 cosine
    sin(2 x 17t) [ 0.75,   0.99,    0.41,   ...,  0.75    ]   ← harmonic 2 sine
    cos(2 x 17t) [ 0.66,  -0.14,  -0.91,   ...,  0.66    ]   ← harmonic 2 cosine

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
    # regularization constant for the covariance matrices; the 'c' at the beginning
    # is to indicate that it is a constant (meaning, its value doesn't change throughout the runtime)
    cREG = 1e-6
    # X shape: (n_channels, n_samples)
    # Y shape: (2*n_harmonics, n_samples)
    X = data_buffer
    Y = reference_signals
    n = X.shape[1] # (9, 250) -> shape[0] = 9, shape[1] = 250 -> returns 250
    
    # Step 1: center X (Y does not need to be centered)
    X = X - X.mean(axis=1, keepdims=True)

    # Step 2: covariance matrices for X 
    Cxx = (X @ X.T) / n + cREG * np.eye(X.shape[0])

    # Step 2(cont'd) & steps 3-5
    rhos = [] # store the rho value of each reference
    for y in Y:
        Cyy = (y @ y.T) / n + cREG * np.eye(y.shape[0]) # step 2
        Cxy = (X @ y.T) / n # step 2
        M = np.linalg.inv(Cxx) @ Cxy @ np.linalg.inv(Cyy) @ Cxy.T
        eigenvalues = np.linalg.eigvals(M)
        eigenvalues = np.real(eigenvalues)           # discard tiny imaginary artifacts
        eigenvalues = np.clip(eigenvalues, 0, None)  # ensure no negatives from numerical noise
        rho = np.sqrt(np.max(eigenvalues))
        rhos.append(rho) 

    return np.argmax(rhos)