# Tests/test_cca.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Python-Processor'))

import numpy as np
import pyxdf
from CCA import perform_CCA, build_reference_signals

class MockDataBuffer:
    def __init__(self, data):
        self.sliding_window = data.T
    
    def ret_sliding_window(self):
        return self.sliding_window
# ── config ────────────────────────────────────────────────────────────────────
XDF_PATH  = os.path.join(os.path.dirname(__file__), '..', 'synapserun_eeg_subj01.xdf')
FREQS     = (7, 11, 13, 17)
FS        = 250
N_SAMPLES = 250  # 1 second

# ── load xdf ──────────────────────────────────────────────────────────────────
data, header = pyxdf.load_xdf(XDF_PATH)
eeg_stream   = data[0]
eeg_data     = np.array(eeg_stream['time_series']).T  # (n_channels, n_samples_total)

print(f"EEG shape: {eeg_data.shape}")

# ── build references ──────────────────────────────────────────────────────────
references = build_reference_signals(FREQS, N_SAMPLES, FS)

# ── test 1: does it run at all? ───────────────────────────────────────────────
print("\n--- Test 1: runs without crashing ---")
segment = eeg_data[:, :N_SAMPLES]
result  = perform_CCA(MockDataBuffer(segment), references, FREQS)
print(f"Result frequency: {result} Hz")
assert result in FREQS, "Result not a valid frequency!"
print("PASSED")

# ── test 2: sanity check on random noise ─────────────────────────────────────
print("\n--- Test 2: random noise returns something valid ---")
noise  = np.random.randn(eeg_data.shape[0], N_SAMPLES)
result = perform_CCA(MockDataBuffer(noise), references, FREQS)
print(f"Result frequency: {result} Hz")
assert result in FREQS, "Result not a valid frequency!"
print("PASSED")

# ── test 3: injected signal returns correct frequency ────────────────────────
print("\n--- Test 3: injected SSVEP signal ---")
t = np.arange(N_SAMPLES) / FS
for freq in FREQS:
    eeg_injected = np.random.randn(eeg_data.shape[0], N_SAMPLES) * 0.5
    eeg_injected += np.sin(2 * np.pi * freq * t)
    result = perform_CCA(MockDataBuffer(eeg_injected), references, FREQS)
    status = "PASSED" if result == freq else "FAILED"
    print(f"{status} — injected {freq}Hz | expected {freq}Hz | got {result}Hz")