"""Run main() of the original .pyc or of the recovered .py on the same cached data.

Needs Python 3.12 (the .pyc is 3.12 bytecode).

Usage (from the repository root):
    python recovery/run_compare.py orig  /tmp/cmp/orig  > /tmp/cmp/orig.txt
    python recovery/run_compare.py src   /tmp/cmp/src   > /tmp/cmp/src.txt
    diff /tmp/cmp/orig.txt /tmp/cmp/src.txt   # only the plot folder path differs

Both runs share the yfinance cache in <OUT_DIR>/../cache, so they see identical prices.
"""
import hashlib, marshal, os, pickle, sys

mode, out_dir = sys.argv[1], os.path.abspath(sys.argv[2])
os.makedirs(out_dir, exist_ok=True)
cache_dir = os.path.join(os.path.dirname(out_dir), 'cache')
os.makedirs(cache_dir, exist_ok=True)

import yfinance as yf
_real = yf.download
def cached_download(*args, **kw):
    key = hashlib.sha1(repr((args, sorted(kw.items()))).encode()).hexdigest()
    path = os.path.join(cache_dir, key + '.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f: return pickle.load(f)
    df = _real(*args, **kw)
    with open(path, 'wb') as f: pickle.dump(df, f)
    return df
yf.download = cached_download

here = os.path.dirname(os.path.abspath(__file__))
if mode == 'orig':
    with open(os.path.join(here, 'crypto_quant_lab.cpython-312.pyc'), 'rb') as f:
        code = marshal.loads(f.read()[16:])
else:
    path = os.path.join(os.path.dirname(here), 'crypto_quant_lab.py')
    code = compile(open(path).read(), path, 'exec')
g = {'__name__': 'crypto_quant_lab'}
exec(code, g)
g['OUT_DIR'] = out_dir
g['main']()
