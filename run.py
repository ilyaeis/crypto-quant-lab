"""Run the full research pipeline (crypto_quant_lab.main) on fresh Yahoo Finance data.

- Plots go to results/figures/ (the original script wrote them to the author's Desktop).
- Every yfinance download is cached in data/cache/, so a second run uses the same data.
  Delete data/cache/ to fetch fresh prices.

Usage:  python run.py > results/run_log.txt
"""
import hashlib
import os
import pickle

import yfinance as yf

import crypto_quant_lab as cql

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'data', 'cache')
FIGURES = os.path.join('results', 'figures')

_download = yf.download


def cached_download(*args, **kwargs):
    key = hashlib.sha1(repr((args, sorted(kwargs.items()))).encode()).hexdigest()
    path = os.path.join(CACHE, key + '.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    df = _download(*args, **kwargs)
    with open(path, 'wb') as f:
        pickle.dump(df, f)
    return df


if __name__ == '__main__':
    os.chdir(ROOT)
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    yf.download = cached_download
    cql.OUT_DIR = FIGURES
    cql.main()
