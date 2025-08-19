import pandas as pd
from dam_monitoring.preprocess.wavelet_iso import extract_high_freq

def test_extract_high_freq_runs():
    s = pd.Series(range(128))
    out = extract_high_freq(s)
    assert len(out) == len(s)
