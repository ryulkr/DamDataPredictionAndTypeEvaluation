from __future__ import annotations
import pandas as pd
import pywt
from sklearn.ensemble import IsolationForest

def extract_high_freq(series: pd.Series, wavelet: str = "db4", level: int = 3) -> pd.Series:
    coeffs = pywt.wavedec(series.values, wavelet=wavelet, level=level)
    high = pywt.waverec([None] + coeffs[1:], wavelet)
    return pd.Series(high[: len(series)], index=series.index)

def detect_outliers_df(df: pd.DataFrame, contamination=0.01, random_state=42) -> pd.Series:
    hf = df.apply(extract_high_freq)
    hf = hf.dropna()
    model = IsolationForest(contamination=contamination, random_state=random_state).fit(hf)
    labels = pd.Series(model.predict(hf) == -1, index=hf.index)  # True = outlier
    return labels.reindex(df.index).fillna(False)
