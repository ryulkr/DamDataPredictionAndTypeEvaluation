import pandas as pd
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest

# helper: high‑frequency component via 3‑level DWT (db4)
def extract_high_freq(series, wavelet='db4', level=3):
    coeffs = pywt.wavedec(series.values, wavelet=wavelet, level=level)
    high_freq = pywt.waverec([None] + coeffs[1:], wavelet=wavelet)
    return pd.Series(high_freq[:len(series)], index=series.index)

# file paths
files = {
    '감포': './GPS통합본(준공이후)_기초자료/감포_GPS 변위계1.csv',
    '남강': './GPS통합본(준공이후)_기초자료/남강_GPS 변위계1.csv',
    '충주': './GPS통합본(준공이후)_기초자료/충주_GPS 변위계1.csv'
}

rows = []

for dam, path in files.items():
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # assume displacement axes columns (all numeric)
    hf = df.apply(extract_high_freq)
    hf_nonan = hf.dropna()
    
    # Isolation Forest on high‑freq
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(hf_nonan)
    labels = iso.predict(hf_nonan)
    
    # full boolean mask
    mask = pd.Series(False, index=df.index)
    mask.loc[hf_nonan.index] = (labels == -1)
    
    # stats before
    mu_before = df.mean()
    std_before = df.std()
    skew_before = df.skew()
    kurt_before = df.kurtosis()
    
    # replace detected outliers with NaN
    df_clean = df.copy()
    df_clean[mask] = np.nan
    
    # stats after
    mu_after = df_clean.mean()
    std_after = df_clean.std()
    skew_after = df_clean.skew()
    kurt_after = df_clean.kurtosis()
    
    # deltas (absolute)
    delta_mu = (mu_after - mu_before).abs().mean()
    delta_std = (std_after - std_before).abs().mean()
    delta_skew = (skew_after - skew_before).abs().mean()
    delta_kurt = (kurt_after - kurt_before).abs().mean()
    
    # relative change of mean vs raw std
    rel_mu = (delta_mu / std_before.mean()) * 100
    
    rows.append({
        'Dam': dam,
        'ΔMean (mm)': round(delta_mu, 3),
        'ΔStd (mm)': round(delta_std, 3),
        'ΔSkewness': round(delta_skew, 3),
        'ΔKurtosis': round(delta_kurt, 3),
        'Δμ / σ_raw (%)': round(rel_mu, 1)
    })

table2 = pd.DataFrame(rows)
print(table2)
