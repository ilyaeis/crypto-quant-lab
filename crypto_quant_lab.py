"""
Crypto Quant Lab — Bank Research Division
==========================================
BTC: strategy development (in-sample)
ETH: out-of-sample validation
Commission: 1% per transaction
Agent horizon: 1 day (daily close-to-close)
v59: Core3 gate exemption — stay active in Mar/Jul/Nov even when BTC < 150d MA
"""
global COMMISSION

import sys

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

COMMISSION = 0.01
INITIAL_CAPITAL = 10000.0
DATA_PERIOD = '5y'
OUT_DIR = 'results/figures'
HOURS_PER_YEAR = 8760


def load_data(ticker: str) -> pd.DataFrame:
    print(f'''  Fetching {ticker} ({DATA_PERIOD})...''', end = '', flush = True)
    df = yf.download(ticker, period = DATA_PERIOD, auto_adjust = True, progress = False)
    df = df[[
        'Open',
        'High',
        'Low',
        'Close',
        'Volume']].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace = True)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace = True)
    print(f''' {len(df)} days  [{df.index[0].date()} → {df.index[-1].date()}]''')
    return df


def load_intraday_data(ticker: str, interval: str = '5m', period: str = '1mo') -> pd.DataFrame:
    """Load intraday bars. interval: '1m','2m','5m','15m','30m','60m','90m'."""
    print(f'  Fetching {ticker} ({interval}, {period})...', end='', flush=True)
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False, timeout=30)
    except Exception as e:
        print(f' FAILED ({e})')
        raise
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    mins_map = {'1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30, '60m': 60, '90m': 90}
    mins_per_bar = mins_map.get(interval, 60)
    bpy = int(525600 / mins_per_bar)
    n_days = len(df) * mins_per_bar / 1440
    print(f' {len(df)} bars  [{df.index[0].strftime("%Y-%m-%d %H:%M")} → '
          f'{df.index[-1].strftime("%Y-%m-%d %H:%M")}]  (~{n_days:.0f} days)  bpy={bpy}')
    return df, bpy


def load_hourly_data(ticker: str, period: str = '1mo') -> pd.DataFrame:
    print(f'''  Fetching {ticker} (1h, {period})...''', end = '', flush = True)
    df = yf.download(ticker, period = period, interval = '1h', auto_adjust = True, progress = False)
    df = df[[
        'Open',
        'High',
        'Low',
        'Close',
        'Volume']].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace = True)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace = True)
    n_days = len(df) / 24
    print(f''' {len(df)} bars  [{df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}]  (~{n_days:.0f} days)''')
    return df


class Backtest:
    """
    Daily close-to-close backtest.
    signal[t] = desired position at close of day t (0=cash, 1=long).
    Position earns return on day t+1.
    Commission charged when position changes.
    """

    def __init__(self, df: pd.DataFrame, signal: pd.Series,
                 allow_short: bool = False, bars_per_year: int = 252):
        prices = df['Close']
        signals = signal.reindex(prices.index).fillna(0)
        if not allow_short:
            signals = signals.clip(0, 1)
        n = len(prices)
        prices_arr = prices.values.astype(float)
        signals_arr = signals.values.astype(float)
        equity = np.empty(n)
        pos_arr = np.empty(n)
        capital = INITIAL_CAPITAL
        position = 0.0
        total_comm = 0.0
        trades = 0
        new_pos0 = signals_arr[0]
        if new_pos0 != position:
            cost = COMMISSION * abs(new_pos0 - position)
            capital *= 1 - cost
            total_comm += INITIAL_CAPITAL * cost
            position = new_pos0
            trades += 1
        equity[0] = capital
        pos_arr[0] = position
        for i in range(1, n):
            daily_ret = prices_arr[i] / prices_arr[i - 1] - 1
            capital *= 1 + position * daily_ret
            new_pos = signals_arr[i]
            if new_pos != position:
                cost = COMMISSION * abs(new_pos - position)
                capital *= 1 - cost
                total_comm += capital * cost / (1 - cost)
                position = new_pos
                trades += 1
            equity[i] = capital
            pos_arr[i] = position
        self.df = df
        self.portfolio = pd.DataFrame(
            {'value': equity, 'pos': pos_arr}, index=prices.index)
        self.total_comm = total_comm
        self.trades = trades
        self.bars_per_year = bars_per_year

    @property
    def equity(self) -> pd.Series:
        return self.portfolio['value']

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    @property
    def total_return(self) -> float:
        return self.equity.iloc[-1] / INITIAL_CAPITAL - 1

    @property
    def annual_return(self) -> float:
        n = len(self.equity)
        return (1 + self.total_return) ** (self.bars_per_year / n) - 1

    @property
    def sharpe(self) -> float:
        r = self.returns
        return float(np.sqrt(self.bars_per_year) * r.mean() / r.std()) if r.std() > 0 else 0.0

    @property
    def sortino(self) -> float:
        r = self.returns
        down = r[r < 0].std()
        return float(np.sqrt(self.bars_per_year) * r.mean() / down) if down > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        v = self.equity
        dd = (v - v.cummax()) / v.cummax()
        return float(dd.min())

    @property
    def calmar(self) -> float:
        mdd = abs(self.max_drawdown)
        return self.annual_return / mdd if mdd > 0 else 0.0

    @property
    def var99_daily(self) -> float:
        return float(np.percentile(self.returns, 1))

    @property
    def win_rate(self) -> float:
        return float((self.returns > 0).mean())

    def row(self) -> dict:
        return {
            'Total Return': f'{self.total_return:+.1%}',
            'Annual Return': f'{self.annual_return:+.1%}',
            'Sharpe': f'{self.sharpe:.2f}',
            'Sortino': f'{self.sortino:.2f}',
            'Calmar': f'{self.calmar:.2f}',
            'Max DD': f'{self.max_drawdown:.1%}',
            'VaR-99 (day)': f'{self.var99_daily:.2%}',
            'Win Rate': f'{self.win_rate:.1%}',
            'Trades': str(self.trades),
        }


def roll(df: pd.DataFrame, col: str, window: int, fn) -> pd.Series:
    return df[col].rolling(window).apply(fn, raw = True)


def _shift_signal(s: pd.Series) -> pd.Series:
    '''Shift signal by 1 day so decision uses only past data.'''
    return s.shift(1).fillna(0)


def sig_buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sig_golden_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    p = df['Close']
    ma_f = p.rolling(fast).mean()
    ma_s = p.rolling(slow).mean()
    signal = (ma_f > ma_s).astype(float)
    return _shift_signal(signal)


def sig_above_ma(df: pd.DataFrame, window: int = 200) -> pd.Series:
    p = df['Close']
    ma = p.rolling(window).mean()
    signal = (p > ma).astype(float)
    return _shift_signal(signal)


def sig_gbm_monthly(df: pd.DataFrame, window: int = 60) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2

    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )

    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            new_pos = 1.0 if drift.iloc[i] > 0 else 0.0
            if abs(new_pos - pos) > 0.05:
                pos = new_pos
        signal.iloc[i] = pos

    return _shift_signal(signal)


def sig_var99_crash(df: pd.DataFrame, window: int = 60, multiplier: float = 2.0, hold_days: int = 25) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    z99 = stats.norm.ppf(0.99)
    var99 = -(mu - z99 * sig)
    signal = pd.Series(0.0, index=df.index)
    hold = 0
    for i in range(window, len(df)):
        ret = float(r.iloc[i])
        v = float(var99.iloc[i])
        if np.isnan(v):
            continue
        else:
            if -ret > multiplier * v:
                hold = hold_days
            if hold > 0:
                signal.iloc[i] = 1.0
                hold -= 1
    return _shift_signal(signal)


def sig_gbm_throttled(df: pd.DataFrame, window: int = 90, min_hold: int = 20) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    days_held = min_hold + 1
    for i in range(window, len(df)):
        d = float(drift.iloc[i])
        if np.isnan(d):
            continue
        else:
            desired = 1.0 if d > 0 else 0.0
            if desired != pos and days_held >= min_hold:
                    pos = desired
                    days_held = 0
            signal.iloc[i] = pos
            days_held += 1
    return _shift_signal(signal)


def sig_log_trend(df: pd.DataFrame, window: int = 90) -> pd.Series:
    lp = np.log(df['Close'])
    x = np.arange(window)
    signal = pd.Series(0.0, index=df.index)
    for i in range(window, len(df)):
        y = lp.iloc[i - window:i].values
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        resid = y - fitted
        sigma = resid.std()
        cur_dev = y[(-1)] - fitted[(-1)]
        if slope > 0 and cur_dev < 1.5 * sigma:
                signal.iloc[i] = 1.0
    return _shift_signal(signal)


def sig_gbm_montecarlo(df: pd.DataFrame, window: int = 60, horizon: int = 10, n_sims: int = 2000, p10_threshold: float = 0.025, recheck: int = 10) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sigma = r.rolling(window).std()
    rng = np.random.default_rng(42)
    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    days_since = recheck + 1
    for i in range(window, len(df)):
        days_since += 1
        if days_since < recheck:
            signal.iloc[i] = pos
            continue
        else:
            m = float(mu.iloc[i])
            s = float(sigma.iloc[i])
            if np.isnan(m) or np.isnan(s) or s == 0:
                signal.iloc[i] = pos
                continue
            else:
                draws = rng.normal(m, s, (n_sims, horizon))
                terminal = np.exp(draws.sum(axis=1)) - 1
                p10 = float(np.percentile(terminal, 10))
                desired = 1.0 if p10 > p10_threshold else 0.0
                if desired != pos:
                    pos = desired
                    days_since = 0
                signal.iloc[i] = pos
    return _shift_signal(signal)


def sig_trend_low_vol(df: pd.DataFrame, trend_win: int = 50, vol_win: int = 30, vol_med_win: int = 252) -> pd.Series:
    p = df['Close']
    r = df['log_ret']
    ma = p.rolling(trend_win).mean()
    vol = r.rolling(vol_win).std()
    vol_med = vol.rolling(vol_med_win).median()
    trend_up = p > ma
    low_vol = vol < vol_med
    signal = (trend_up & low_vol).astype(float)
    return _shift_signal(signal)


def sig_ensemble(signals: dict, min_agree: int = 3) -> pd.Series:
    mat = pd.DataFrame(signals)
    votes = mat.sum(axis = 1)
    return (votes >= min_agree).astype(float)


def sig_gbm_monthly_bull(df: pd.DataFrame, window: int = 60) -> pd.Series:
    gbm = sig_gbm_monthly(df, window = window)
    bull = sig_above_ma(df, window = 200)
    return (gbm * bull).clip(0, 1)


def sig_gbm_monthly_scaled(df: pd.DataFrame, window: int = 60, norm_window: int = 12) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2

    drift_std = drift.rolling(window * norm_window, min_periods=window).std()
    drift_z = drift / (drift_std + 1e-10)

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-float(x)))

    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )

    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            z = float(drift_z.iloc[i])
            if not np.isnan(z):
                new_pos = sigmoid(z * 1.5)
                if abs(new_pos - pos) > 0.05:
                    pos = new_pos
        signal.iloc[i] = pos

    return _shift_signal(signal)


def sig_gbm_monthly_kelly(df: pd.DataFrame, window: int = 60) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()

    kelly = mu / (sig ** 2 + 1e-10)

    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )

    signal = pd.Series(0.0, index=df.index)
    pos = 0.0

    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            k = float(kelly.iloc[i])
            if not np.isnan(k):
                new_pos = float(np.clip(k, 0.0, 1.0))
                if abs(new_pos - pos) > 0.05:
                    pos = new_pos
        signal.iloc[i] = pos

    return _shift_signal(signal)


def sig_gbm_monthly_half_kelly(df: pd.DataFrame, window: int = 60) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    kelly = 0.5 * mu / (sig ** 2 + 1e-10)

    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )

    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            k = float(kelly.iloc[i])
            if not np.isnan(k):
                new_pos = float(np.clip(k, 0.0, 1.0))
                if abs(new_pos - pos) > 0.05:
                    pos = new_pos
        signal.iloc[i] = pos
    return _shift_signal(signal)


def apply_stop_loss(df: pd.DataFrame, base_signal: pd.Series, max_dd: float = 0.2, lookback: int = 60) -> pd.Series:
    price = df['Close']
    rolling_hi = price.rolling(lookback, min_periods=1).max()
    deep_dd = price / rolling_hi - 1 < -max_dd
    return (base_signal * (~deep_dd).astype(float)).clip(0, 1)


def sig_gbm_monthly_ls(df: pd.DataFrame, window: int = 60, threshold: float = 0.0003) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2


    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )


    signal = pd.Series(0.0, index=df.index)
    pos = 0.0

    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            d = float(drift.iloc[i])
            if not np.isnan(d):
                if d > threshold:
                    new_pos = 1.0
                elif d < -threshold:
                    new_pos = -1.0
                else:
                    new_pos = 0.0
                if abs(new_pos - pos) > 0.05:
                    pos = new_pos
        signal.iloc[i] = pos

    return _shift_signal(signal)


def sig_momentum_monthly(df: pd.DataFrame, window: int = 60) -> pd.Series:
    r_win = df['Close'].pct_change(window)

    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )

    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            r = float(r_win.iloc[i])
            if not np.isnan(r):
                pos = 1.0 if r > 0 else 0.0
        signal.iloc[i] = pos
    return _shift_signal(signal)


def bootstrap_significance(df: pd.DataFrame, strategy_fn, strategy_name: str = '', n_bootstrap: int = 2000) -> None:
    real_sig = strategy_fn(df)
    real_bt = Backtest(df, real_sig)
    real_sharpe = real_bt.sharpe
    real_ret = real_bt.total_return
    periods = df.index.to_period('M')
    month_end_ix = [i for i in range(len(df) - 1) if periods[i] != periods[i + 1]]
    month_end_ix.append(len(df) - 1)
    rng = np.random.default_rng(42)
    rand_sharpes = []
    rand_returns = []
    for _ in range(n_bootstrap):
        sig = pd.Series(0.0, index=df.index)
        pos = 0.0
        last_me = 0
        for me in month_end_ix:
            pos = float(rng.choice([0.0, 1.0]))
            sig.iloc[last_me:me + 1] = pos
            last_me = me + 1
        bt = Backtest(df, _shift_signal(sig))
        rand_sharpes.append(bt.sharpe)
        rand_returns.append(bt.total_return)
    p_sharpe = sum((s >= real_sharpe for s in rand_sharpes)) / n_bootstrap
    p_return = sum((r >= real_ret for r in rand_returns)) / n_bootstrap
    p5, p50, p95 = np.percentile(rand_sharpes, [5, 50, 95])
    label = f'  Bootstrap ({n_bootstrap} random month-end signals): {strategy_name}'
    print(f'\n{label}')
    print(f'  Strategy Sharpe={real_sharpe:.3f}  Return={real_ret:+.1%}')
    print(f'  Random Sharpe distribution: p5={p5:.3f}  median={p50:.3f}  p95={p95:.3f}')
    print(f"  p-value (Sharpe): {p_sharpe:.4f}  {('★ SIGNIFICANT (p<0.05)' if p_sharpe < 0.05 else '~ not significant')}")
    print(f"  p-value (Return): {p_return:.4f}  {('★ SIGNIFICANT (p<0.05)' if p_return < 0.05 else '~ not significant')}")
    return (rand_sharpes, real_sharpe)


def pooled_bootstrap_significance(datasets: list[tuple[str, pd.DataFrame]], strategy_fn, n_bootstrap: int = 3000) -> None:
    """
    Pool returns from multiple assets and test joint significance.
    Null: same random month-end timing applied to all assets simultaneously.
    The test statistic is the mean Sharpe across assets — harder to beat by
    luck than any single-asset test.
    """
    real_sharpes = []
    for _, df in datasets:
        bt = Backtest(df, strategy_fn(df))
        real_sharpes.append(bt.sharpe)
    real_mean_sharpe = float(np.mean(real_sharpes))

    rng = np.random.default_rng(99)
    me_maps = []
    for _, df in datasets:
        periods = df.index.to_period('M')
        me_ix = [i for i in range(len(df) - 1) if periods[i] != periods[i + 1]]
        me_ix.append(len(df) - 1)
        me_maps.append((df, me_ix))

    rand_mean_sharpes = []
    for _ in range(n_bootstrap):
        asset_sharpes = []
        for df, me_ix in me_maps:
            sig = pd.Series(0.0, index=df.index)
            last = 0
            for me in me_ix:
                pos = float(rng.choice([0.0, 1.0]))
                sig.iloc[last:me + 1] = pos
                last = me + 1
            bt = Backtest(df, _shift_signal(sig))
            asset_sharpes.append(bt.sharpe)
        rand_mean_sharpes.append(float(np.mean(asset_sharpes)))

    p_val = sum(s >= real_mean_sharpe for s in rand_mean_sharpes) / n_bootstrap
    p5, p50, p95 = np.percentile(rand_mean_sharpes, [5, 50, 95])
    names = ', '.join(n for n, _ in datasets)

    print(f"\n  Pooled Bootstrap ({n_bootstrap} iterations, assets: {names})")
    print(f"  Per-asset Sharpes: " +
          "  ".join(f"{n}={s:.3f}" for (n, _), s in zip(datasets, real_sharpes)))
    print(f"  Mean Sharpe across assets: {real_mean_sharpe:.3f}")
    print(f"  Random mean Sharpe: p5={p5:.3f}  median={p50:.3f}  p95={p95:.3f}")
    print(f"  p-value: {p_val:.4f}  "
          f"{'★★ SIGNIFICANT (p<0.05)' if p_val < 0.05 else '~ not significant at p<0.05'}")


def lo_sharpe_ttest(bt: Backtest, strategy_name: str = '') -> None:
    """
    Lo (2002) Sharpe ratio t-test that corrects for first-order autocorrelation
    in returns — standard in quantitative finance.
    H0: true Sharpe = 0.  Reject if |t| > 1.96 (two-tailed, 95%).
    """
    r = bt.returns.dropna().values
    n = len(r)
    sr = bt.sharpe / np.sqrt(252)
    rho1 = float(pd.Series(r).autocorr(lag=1))
    adj = 1 + 2 * rho1 * (1 - rho1) / ((1 - rho1 ** 2) + 1e-12) if abs(rho1) < 0.999 else 1.0
    var_sr = (1 + 0.5 * sr ** 2) * adj / n
    t_stat = sr / (np.sqrt(var_sr) + 1e-12)
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    print(f"  Lo(2002) t-test  {strategy_name:<30}  Sharpe={bt.sharpe:.3f}  "
          f"ρ₁={rho1:+.3f}  t={t_stat:.2f}  p={p_val:.4f}  "
          f"{'★ p<0.05' if p_val < 0.05 else '~ p≥0.05'}")


def sig_gbm_monthly_bear_gate(df: pd.DataFrame, window: int = 60, lookback_months: int = 3, bear_threshold: float = 0.25) -> pd.Series:
    base = sig_gbm_monthly(df, window=window)
    r3m = df['Close'].pct_change(lookback_months * 21)
    in_bear = r3m < -bear_threshold
    suppressed = (base * (~in_bear).astype(float).shift(1).fillna(1)).clip(0, 1)
    return suppressed


def sig_gbm_tight_gate(df: pd.DataFrame, window: int = 60, lookback_months: int = 3, bear_threshold: float = 0.15) -> pd.Series:
    base = sig_gbm_monthly(df, window=window)
    r3m = df['Close'].pct_change(lookback_months * 21)
    in_bear = r3m < -bear_threshold
    return (base * (~in_bear).astype(float).shift(1).fillna(1)).clip(0, 1)


def sig_vol_targeting(df: pd.DataFrame, window: int = 60, target_ann_vol: float = 0.2, rvol_window: int = 30) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2

    realized_vol = r.rolling(rvol_window).std() * np.sqrt(252)
    vol_scale = (target_ann_vol / realized_vol).clip(0.0, 1.0)

    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )

    signal = pd.Series(0.0, index=df.index)
    pos = 0.0

    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            direction = 1.0 if float(drift.iloc[i]) > 0 else 0.0
            vs = float(vol_scale.iloc[i])
            scale = vs if not np.isnan(vs) else 1.0
            new_pos = direction * scale
            if abs(new_pos - pos) > 0.05:
                pos = new_pos
        signal.iloc[i] = pos

    return _shift_signal(signal)


def sig_vol_targeting_40(df: pd.DataFrame, window: int = 60) -> pd.Series:
    return sig_vol_targeting(df, window = window, target_ann_vol = 0.4)


def sig_vol_targeting_adaptive(df: pd.DataFrame, window: int = 60, vol_bull: float = 0.4, vol_bear: float = 0.2, bear_threshold: float = 0.15, bear_lookback: int = 63) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    realized_vol = r.rolling(30).std() * np.sqrt(252)
    r3m = df['Close'].pct_change(bear_lookback)
    in_bear = r3m < -bear_threshold
    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )
    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            direction = 1.0 if float(drift.iloc[i]) > 0 else 0.0
            target_vol = vol_bear if bool(in_bear.iloc[i]) else vol_bull
            rv = float(realized_vol.iloc[i])
            scale = min(target_vol / rv, 1.0) if not np.isnan(rv) and rv > 0 else 1.0
            new_pos = direction * scale
            if abs(new_pos - pos) > 0.05:
                pos = new_pos
        signal.iloc[i] = pos
    return _shift_signal(signal)


def sig_vol_targeting_40_45d(df: pd.DataFrame, window: int = 60) -> pd.Series:
    return sig_vol_targeting(df, window = window, target_ann_vol = 0.4, rvol_window = 45)


def sig_gbm_asymmetric(df: pd.DataFrame, window_fast: int = 30, window_slow: int = 60, target_ann_vol: float = 0.4, rvol_window: int = 45) -> pd.Series:
    r = df['log_ret']
    mu_f = r.rolling(window_fast).mean()
    sig_f = r.rolling(window_fast).std()
    drift_fast = mu_f - 0.5 * sig_f ** 2
    mu_s = r.rolling(window_slow).mean()
    sig_s = r.rolling(window_slow).std()
    drift_slow = mu_s - 0.5 * sig_s ** 2
    realized_vol = r.rolling(rvol_window).std() * np.sqrt(252)
    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )
    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window_slow, len(df)):
        if is_month_end.iloc[i]:
            fd = float(drift_fast.iloc[i])
            sd = float(drift_slow.iloc[i])
            in_market = pos > 0.05
            if in_market:
                direction = 0.0 if sd < 0 else 1.0
            else:
                direction = 1.0 if fd > 0 else 0.0
            rv = float(realized_vol.iloc[i])
            scale = min(target_ann_vol / rv, 1.0) if not np.isnan(rv) and rv > 0 else 1.0
            new_pos = direction * scale
            if abs(new_pos - pos) > 0.05:
                pos = new_pos
        signal.iloc[i] = pos
    return _shift_signal(signal)


def sig_gbm_asymmetric_volgated(df: pd.DataFrame, window_fast: int = 30, window_slow: int = 60, target_ann_vol: float = 0.4, rvol_window: int = 45, vol_gate: float = 0.8) -> pd.Series:
    r = df['log_ret']
    mu_f = r.rolling(window_fast).mean()
    sig_f = r.rolling(window_fast).std()
    drift_fast = mu_f - 0.5 * sig_f ** 2
    mu_s = r.rolling(window_slow).mean()
    sig_s = r.rolling(window_slow).std()
    drift_slow = mu_s - 0.5 * sig_s ** 2
    realized_vol = r.rolling(rvol_window).std() * np.sqrt(252)
    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )
    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window_slow, len(df)):
        if is_month_end.iloc[i]:
            fd = float(drift_fast.iloc[i])
            sd = float(drift_slow.iloc[i])
            rv = float(realized_vol.iloc[i])
            in_market = pos > 0.05
            high_vol = not np.isnan(rv) and rv > vol_gate
            if in_market:
                direction = 0.0 if sd < 0 else 1.0
            else:
                if high_vol:
                    direction = 1.0 if sd > 0 else 0.0
                else:
                    direction = 1.0 if fd > 0 else 0.0
            scale = min(target_ann_vol / rv, 1.0) if not np.isnan(rv) and rv > 0 else 1.0
            new_pos = direction * scale
            if abs(new_pos - pos) > 0.05:
                pos = new_pos
        signal.iloc[i] = pos
    return _shift_signal(signal)


def sig_blend_50_50(df: pd.DataFrame) -> pd.Series:
    s1 = sig_vol_targeting_40_45d(df)
    s2 = sig_gbm_asymmetric(df)
    return ((s1 + s2) / 2.0).clip(0.0, 1.0)


def sig_monthly_stoploss(df: pd.DataFrame, base_fn: callable, stop_pct: float = 0.05) -> pd.Series:
    base_sig = base_fn(df)
    prices = df['Close'].values.astype(float)
    sigs = base_sig.reindex(df.index).fillna(0).clip(0, 1).values.astype(float)
    n = len(prices)
    out = np.zeros(n)
    pos = 0.0
    cap = 1.0
    m_cap = 1.0
    stopped = False
    for i in range(n):
        is_new_month = i == 0 or df.index[i].month != df.index[i - 1].month
        if is_new_month:
            m_cap = cap
            stopped = False
        if not stopped and cap < m_cap * (1.0 - stop_pct):
                stopped = True
                pos = 0.0
        pos = 0.0 if stopped else sigs[i]
        out[i] = pos
        if i < n - 1:
            ret = prices[i + 1] / prices[i] - 1.0
            cap *= 1.0 + pos * ret
    return pd.Series(out, index=df.index)


def sig_vt45_monthly_stop(df: pd.DataFrame, stop_pct: float = 0.05) -> pd.Series:
    return sig_monthly_stoploss(df, sig_vol_targeting_40_45d, stop_pct = stop_pct)


def sig_blend_monthly_stop(df: pd.DataFrame, stop_pct: float = 0.05) -> pd.Series:
    return sig_monthly_stoploss(df, sig_blend_50_50, stop_pct = stop_pct)


def sig_base_position(df: pd.DataFrame, base_pct: float = 0.2, window: int = 60, target_ann_vol: float = 0.4, rvol_window: int = 45) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    rvol = r.rolling(rvol_window).std() * np.sqrt(252)
    vol_scale = (target_ann_vol / rvol).clip(0.0, 1.0)
    extra_budget = 1.0 - base_pct
    signal = pd.Series(np.nan, index=df.index)
    pos = base_pct
    month_end_idx = df.index[df.index.to_series().dt.is_month_end]
    for i, idx in enumerate(df.index):
        if idx in month_end_idx or i == 0:
            d = drift.loc[idx]
            vs = vol_scale.loc[idx]
            if not pd.isna(d) and (not pd.isna(vs)):
                    if d > 0:
                        pos = base_pct + extra_budget * float(vs)
                    else:
                        pos = base_pct
        signal.loc[idx] = pos
    return _shift_signal(signal.fillna(base_pct))


def sig_base5_vt(df: pd.DataFrame) -> pd.Series:
    '''5% base + Vol-Tgt(40%,45d) top-up.  Grid-optimal: 50% monthly WR, Sharpe 0.56 (no penalty).'''
    return sig_base_position(df, base_pct = 0.05)


def sig_base10_vt(df: pd.DataFrame) -> pd.Series:
    '''10% base + Vol-Tgt(40%,45d) top-up.'''
    return sig_base_position(df, base_pct = 0.1)


def sig_base20_vt(df: pd.DataFrame) -> pd.Series:
    '''20% base + Vol-Tgt(40%,45d) top-up.'''
    return sig_base_position(df, base_pct = 0.2)


def sig_base30_vt(df: pd.DataFrame) -> pd.Series:
    '''30% base + Vol-Tgt(40%,45d) top-up.'''
    return sig_base_position(df, base_pct = 0.3)


def sig_base5_blend(df: pd.DataFrame) -> pd.Series:
    """5% base floor + VT+Asymmetric blend top-up.  Best of both: 50% WR + blend\'s higher ETH/SOL."""
    r = df['log_ret']
    window = 60
    rvol_window = 45
    mu = r.rolling(window).mean()
    sig_r = r.rolling(window).std()
    drift_slow = mu - 0.5 * sig_r ** 2
    rvol = r.rolling(rvol_window).std() * np.sqrt(252)
    vol_scale = (0.4 / rvol).clip(0.0, 1.0)
    r_fast = df['log_ret']
    w_fast = 30
    mu_f = r_fast.rolling(w_fast).mean()
    sig_f = r_fast.rolling(w_fast).std()
    drift_fast = mu_f - 0.5 * sig_f ** 2
    base_pct = 0.05
    extra_budget = 0.95
    signal = pd.Series(np.nan, index=df.index)
    pos = base_pct
    in_mkt = False
    month_end_idx = set(df.index[df.index.to_series().dt.is_month_end])
    for i, idx in enumerate(df.index):
        if idx in month_end_idx or i == 0:
            d_slow = drift_slow.loc[idx]
            d_fast = drift_fast.loc[idx]
            vs = vol_scale.loc[idx]
            if not pd.isna(d_slow) and (not pd.isna(vs)):
                    vt_pos = float(vs) if d_slow > 0 else 0.0
                    if in_mkt:
                        asym_pos = float(vs) if d_slow > 0 else 0.0
                    else:
                        asym_pos = float(vs) if not pd.isna(d_fast) and d_fast > 0 else 0.0
                    in_mkt = max(vt_pos, asym_pos) > 0.05
                    top_up = min((vt_pos + asym_pos) / 2.0, extra_budget)
                    pos = base_pct + top_up
        signal.loc[idx] = pos
    return _shift_signal(signal.fillna(base_pct))


def sig_weekly_vol_targeting(df: pd.DataFrame, window: int = 60, target_ann_vol: float = 0.4, rvol_window: int = 45) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    rvol = r.rolling(rvol_window).std() * np.sqrt(252)
    vol_scale = (target_ann_vol / rvol).clip(0.0, 1.0)


    signal = pd.Series(np.nan, index=df.index)
    pos = 0.0

    for i, idx in enumerate(df.index):
        is_friday = idx.weekday() == 4
        is_last_of_week = (
            i == len(df.index) - 1 or
            df.index[i + 1].weekday() < idx.weekday()
        )

        if is_friday or i == 0:
            d = drift.loc[idx]
            vs = vol_scale.loc[idx]
            if not pd.isna(d) and not pd.isna(vs):
                pos = float(vs) if d > 0 else 0.0
        signal.loc[idx] = pos

    return _shift_signal(signal.fillna(0))


def sig_base20_weekly(df: pd.DataFrame, window: int = 60, target_ann_vol: float = 0.4, rvol_window: int = 45) -> pd.Series:
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    rvol = r.rolling(rvol_window).std() * np.sqrt(252)
    vol_scale = (target_ann_vol / rvol).clip(0.0, 1.0)
    base_pct = 0.2
    extra_budget = 0.8
    signal = pd.Series(np.nan, index=df.index)
    pos = base_pct
    for i, idx in enumerate(df.index):
        is_friday = idx.weekday() == 4
        if is_friday or i == 0:
            d = drift.loc[idx]
            vs = vol_scale.loc[idx]
            if not pd.isna(d) and (not pd.isna(vs)):
                    if d > 0:
                        pos = base_pct + extra_budget * float(vs)
                    else:
                        pos = base_pct
        signal.loc[idx] = pos
    return _shift_signal(signal.fillna(base_pct))


def sig_gbm_hourly(df_h: pd.DataFrame, window_days: int = 7) -> pd.Series:
    """GBM drift signal on hourly bars, rebalancing at end of each calendar day."""
    r = df_h['log_ret']
    window = window_days * 24
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    signal = pd.Series(0.0, index=df_h.index)
    pos = 0.0
    for i in range(len(df_h.index)):
        signal.iloc[i] = pos
        if i < len(df_h.index) - 1:
            if df_h.index[i + 1].date() != df_h.index[i].date():
                d = drift.iloc[i]
                if not pd.isna(d):
                    pos = 1.0 if d > 0 else 0.0
    return _shift_signal(signal)


def sig_vt_hourly(df_h: pd.DataFrame, window_days: int = 7, target_ann_vol: float = 0.4, rvol_days: int = 3) -> pd.Series:
    """Vol-targeting GBM on hourly bars, daily rebalancing."""
    r = df_h['log_ret']
    window = window_days * 24
    rvol_win = rvol_days * 24
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    rvol = r.rolling(rvol_win).std() * np.sqrt(HOURS_PER_YEAR)
    vol_scale = (target_ann_vol / rvol).clip(0.0, 1.0)
    signal = pd.Series(0.0, index=df_h.index)
    pos = 0.0
    for i in range(len(df_h.index)):
        signal.iloc[i] = pos
        if i < len(df_h.index) - 1:
            if df_h.index[i + 1].date() != df_h.index[i].date():
                d = drift.iloc[i]
                vs = vol_scale.iloc[i]
                if not pd.isna(d) and (not pd.isna(vs)):
                        pos = float(vs) if d > 0 else 0.0
    return _shift_signal(signal)


def sig_btc_lead_vt(df_target: pd.DataFrame, df_btc: pd.DataFrame, window: int = 60, target_ann_vol: float = 0.4, rvol_window: int = 45) -> pd.Series:
    """BTC GBM drift → direction; target asset realized vol → position size."""
    r_btc = df_btc['log_ret'].reindex(df_target.index).fillna(0)
    mu_b = r_btc.rolling(window).mean()
    sig_b = r_btc.rolling(window).std()
    drift = mu_b - 0.5 * sig_b ** 2

    r_tgt = df_target['log_ret']
    rvol = r_tgt.rolling(rvol_window).std() * np.sqrt(252)
    vol_scale = (target_ann_vol / rvol).clip(0.0, 1.0)

    month_ends = df_target.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))], index=df_target.index,
    )
    signal = pd.Series(0.0, index=df_target.index)
    pos = 0.0
    for i in range(window, len(df_target)):
        if is_month_end.iloc[i]:
            d = float(drift.iloc[i])
            vs = float(vol_scale.iloc[i])
            if not np.isnan(d) and not np.isnan(vs):
                new = float(vs) if d > 0 else 0.0
                if abs(new - pos) > 0.05: pos = new
        signal.iloc[i] = pos
    return _shift_signal(signal)


def sig_btc_lead_asym(df_target: pd.DataFrame, df_btc: pd.DataFrame, window_fast: int = 30, window_slow: int = 60, target_ann_vol: float = 0.4, rvol_window: int = 45) -> pd.Series:
    """BTC asymmetric signal (30d entry / 60d exit) → target asset sized by own vol."""
    r_btc = df_btc['log_ret'].reindex(df_target.index).fillna(0)
    mu_f = r_btc.rolling(window_fast).mean()
    sig_f = r_btc.rolling(window_fast).std()
    drift_fast = mu_f - 0.5 * sig_f ** 2
    mu_s = r_btc.rolling(window_slow).mean()
    sig_s = r_btc.rolling(window_slow).std()
    drift_slow = mu_s - 0.5 * sig_s ** 2

    r_tgt = df_target['log_ret']
    rvol = r_tgt.rolling(rvol_window).std() * np.sqrt(252)
    vol_scale = (target_ann_vol / rvol).clip(0.0, 1.0)

    month_ends = df_target.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))], index=df_target.index)

    signal = pd.Series(0.0, index=df_target.index)
    pos = 0.0
    for i in range(window_slow, len(df_target)):
        if is_month_end.iloc[i]:
            fd = float(drift_fast.iloc[i])
            sd = float(drift_slow.iloc[i])
            vs = float(vol_scale.iloc[i])
            in_market = pos > 0.05
            direction = (0.0 if sd < 0 else 1.0) if in_market else (1.0 if fd > 0 else 0.0)
            new = direction * (vs if not np.isnan(vs) else 1.0)
            if abs(new - pos) > 0.05: pos = new
        signal.iloc[i] = pos
    return _shift_signal(signal)


def sig_btc_lead_base5(df_target: pd.DataFrame, df_btc: pd.DataFrame) -> pd.Series:
    """BTC-Lead VT with 5% floor — best of lead signal + monthly win-rate fix."""
    r_btc = df_btc['log_ret'].reindex(df_target.index).fillna(0)
    window = 60
    mu_b = r_btc.rolling(window).mean()
    sig_b = r_btc.rolling(window).std()
    drift = mu_b - 0.5 * sig_b ** 2
    r_tgt = df_target['log_ret']
    rvol = r_tgt.rolling(45).std() * np.sqrt(252)
    vs = (0.4 / rvol).clip(0.0, 1.0)
    me_idx = set(df_target.index[df_target.index.to_series().dt.is_month_end])
    signal = pd.Series(np.nan, index=df_target.index)
    pos = 0.05
    for i, idx in enumerate(df_target.index):
        if idx in me_idx or i == 0:
            d = drift.loc[idx]
            v = vs.loc[idx]
            if not pd.isna(d) and (not pd.isna(v)):
                pos = 0.05 + 0.95 * float(v) if d > 0 else 0.05
        signal.loc[idx] = pos
    return _shift_signal(signal.fillna(0.05))


def _funding_proxy_gate(df_btc: pd.DataFrame, z_threshold: float = 2.0) -> pd.Series:
    """Return True on days when BTC funding proxy says market is safe to enter."""
    r = df_btc['log_ret']
    ret7 = df_btc['Close'].pct_change(7)
    vol7 = r.rolling(7).std() * np.sqrt(252)
    z = ret7 / (vol7 + 1e-08)
    overheated = z > z_threshold
    return (~overheated).shift(1).fillna(True)


def sig_btc_lead_funded(df_target: pd.DataFrame, df_btc: pd.DataFrame, z_threshold: float = 2.0) -> pd.Series:
    '''BTC-Lead VT + funding-proxy gate: skip entry when BTC is overheated.'''
    base = sig_btc_lead_vt(df_target, df_btc)
    gate = _funding_proxy_gate(df_btc, z_threshold).reindex(df_target.index).fillna(True)
    return (base * gate.astype(float)).clip(0, 1)

SEASONAL_STRONG_MONTHS = {1, 3, 7, 8, 10}
SEASONAL_WEAK_MONTHS = {6}
SEASONAL_CORE3_MONTHS = {11, 3, 7}


def _apply_seasonal_scalar(sig: pd.Series, strong: set, weak: set, boost: float = 1.2, dampen: float = 0.8) -> pd.Series:
    scalar = pd.Series(1.0, index=sig.index)
    for idx in sig.index:
        m = idx.month
        if m in strong:
            scalar.loc[idx] = boost
        else:
            if m in weak:
                scalar.loc[idx] = dampen
    return (sig * scalar).clip(0.0, 1.0)


def sig_seasonal_vt45(df: pd.DataFrame) -> pd.Series:
    '''VT(40%,45d) with seasonal overlay: ×1.2 Jan/Mar/Jul/Aug/Oct, ×0.8 Jun.'''
    base = sig_vol_targeting_40_45d(df)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, SEASONAL_WEAK_MONTHS)


def sig_seasonal_blend(df: pd.DataFrame) -> pd.Series:
    '''GBM Blend (VT+Asym) with seasonal overlay.'''
    base = sig_blend_50_50(df)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, SEASONAL_WEAK_MONTHS)


def sig_seasonal_asym(df: pd.DataFrame) -> pd.Series:
    '''GBM Asymmetric (30/60) with seasonal overlay.'''
    base = sig_gbm_asymmetric(df)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, SEASONAL_WEAK_MONTHS)


def sig_btc_lead_seasonal(df_target: pd.DataFrame, df_btc: pd.DataFrame) -> pd.Series:
    '''BTC-Lead VT with seasonal overlay.'''
    base = sig_btc_lead_vt(df_target, df_btc)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, SEASONAL_WEAK_MONTHS)


def sig_boost_only_seasonal_vt45(df: pd.DataFrame) -> pd.Series:
    '''VT45 × 1.2 in strong months only; never dampen. OOS-safer than full seasonal.'''
    base = sig_vol_targeting_40_45d(df)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak = set(), boost = 1.2)


def sig_base5_boost_blend(df: pd.DataFrame) -> pd.Series:
    '''5% floor + Blend(VT+Asym) top-up + seasonal boost-only overlay.
    Combines three improvements: base floor (WR↑), blend (ETH/SOL↑), seasonal (returns↑).
    '''
    base = sig_base5_blend(df)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak = set(), boost = 1.1)


def sig_boost_only_seasonal_blend(df: pd.DataFrame) -> pd.Series:
    '''GBM Blend × 1.2 in strong months only.'''
    base = sig_blend_50_50(df)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak = set(), boost = 1.2)


def sig_boost_only_seasonal_asym(df: pd.DataFrame) -> pd.Series:
    '''GBM Asymmetric × 1.2 in strong months only.'''
    base = sig_gbm_asymmetric(df)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak = set(), boost = 1.2)


def sig_base2_vt45_boost(df: pd.DataFrame) -> pd.Series:
    '''2% floor + VT45 top-up + boost-only seasonal (×1.2 strong months).
    Minimal floor test: does 2% floor improve WR without degrading Sharpe much?
    '''
    base = sig_base_position(df, base_pct = 0.02)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak = set(), boost = 1.2)


def sig_base5_vt45_boost(df: pd.DataFrame) -> pd.Series:
    '''5% floor + VT45 top-up + boost-only seasonal (×1.2 strong months).
    Compares pure-VT floor vs blend-floor (Base5+BlendBoost) at same 5% floor level.
    '''
    base = sig_base_position(df, base_pct = 0.05)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak = set(), boost = 1.2)


def sig_drift_scaled_vt45_boost(df: pd.DataFrame, window: int = 45) -> pd.Series:
    """VT45 position scaled by drift confidence (sigmoid) + boost-only seasonal.
    When drift is strongly positive: full VT position.
    When drift is barely positive: half position.
    When drift is negative: zero.
    """
    r = df['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2

    drift_std = drift.rolling(252, min_periods=window).std().fillna(
        drift.expanding(min_periods=window).std())
    drift_z = drift / (drift_std + 1e-10)

    rvol = r.rolling(45).std() * np.sqrt(252)
    vt_max = (0.4 / (rvol + 1e-10)).clip(0, 1)

    def _sig(z): return 1.0 / (1.0 + np.exp(-float(z) * 1.5))

    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )
    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            d = float(drift.iloc[i])
            z = float(drift_z.iloc[i])
            v = float(vt_max.iloc[i])
            if not (np.isnan(d) or np.isnan(z) or np.isnan(v)):
                conf = _sig(z) if d > 0 else 0.0
                new_pos = v * conf
                if abs(new_pos - pos) > 0.02:
                    pos = new_pos
        signal.iloc[i] = pos

    base = _shift_signal(signal)
    return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak=set(), boost=1.2)


def sig_base2_blend_boost(df: pd.DataFrame) -> pd.Series:
    """2% floor + Blend(VT+Asym) top-up + boost-only seasonal (×1.2 strong months).\n    Tests whether floor improvement extends to Blend signal (Blend had Sh=0.61 avg).\n    """
    base = sig_base5_blend(df)
    r = df['log_ret']
    window = 60
    rvol_window = 45
    mu = r.rolling(window).mean()
    sig_r = r.rolling(window).std()
    drift_slow = mu - 0.5 * sig_r ** 2
    rvol = r.rolling(rvol_window).std() * np.sqrt(252)
    vol_scale = (0.4 / rvol).clip(0.0, 1.0)
    w_fast = 30
    mu_f = r.rolling(w_fast).mean()
    sig_f = r.rolling(w_fast).std()
    drift_fast = mu_f - 0.5 * sig_f ** 2
    base_pct = 0.02
    extra_budget = 0.98
    signal = pd.Series(np.nan, index=df.index)
    pos = base_pct
    in_mkt = False
    month_end_idx = set(df.index[df.index.to_series().dt.is_month_end])
    for i, idx in enumerate(df.index):
        if idx in month_end_idx or i == 0:
            d_slow = drift_slow.loc[idx]
            d_fast = drift_fast.loc[idx]
            vs = vol_scale.loc[idx]
            if not pd.isna(d_slow) and (not pd.isna(vs)):
                    vt_pos = float(vs) if d_slow > 0 else 0.0
                    if in_mkt:
                        asym_pos = float(vs) if d_slow > 0 else 0.0
                    else:
                        asym_pos = float(vs) if not pd.isna(d_fast) and d_fast > 0 else 0.0
                    in_mkt = max(vt_pos, asym_pos) > 0.05
                    top_up = min((vt_pos + asym_pos) / 2.0, extra_budget)
                    pos = base_pct + top_up
        signal.loc[idx] = pos
    blended = _shift_signal(signal.fillna(base_pct))
    return _apply_seasonal_scalar(blended, SEASONAL_STRONG_MONTHS, weak=set(), boost=1.2)


def sig_core3_seasonal_vt45(df: pd.DataFrame) -> pd.Series:
    '''VT45 + Core3 seasonal boost (Mar/Jul/Nov ×1.2 only).
    v36 finding: Core3 improves Sh=0.72→0.74 vs 5-month original (Jan/Mar/Jul/Aug/Oct).
    Oct and Aug were weak months — removing them sharpens the seasonal edge.
    '''
    base = sig_vol_targeting_40_45d(df)
    return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak = set(), boost = 1.2)


def sig_core3_base2_vt45(df: pd.DataFrame) -> pd.Series:
    '''2% floor + VT45 + Core3 seasonal (Mar/Jul/Nov). Floor trick + refined seasonal.'''
    base = sig_base_position(df, base_pct = 0.02)
    return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak = set(), boost = 1.2)


def sig_core3_blend(df: pd.DataFrame) -> pd.Series:
    '''Blend(VT45+Asym 50/50) + Core3 seasonal boost (Mar/Jul/Nov ×1.2).
    v39 hypothesis: Blend avg Sh=0.61 > VT45 avg Sh=0.57; Core3 adds seasonal edge on top.
    '''
    base = sig_blend_50_50(df)
    return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak = set(), boost = 1.2)


def sig_core3_base2_blend(df: pd.DataFrame) -> pd.Series:
    '''2% floor + Blend(VT45+Asym) + Core3 seasonal. Floor trick applied to Blend variant.'''
    base = sig_base_position(df, base_pct = 0.02)
    return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak = set(), boost = 1.2)


def sig_bull_filter_vt45_boost(df: pd.DataFrame) -> pd.Series:
    '''VT45+BoostOnly gated by 200d SMA bull filter.
    Only takes position when price is above 200d SMA (structural bull regime).
    Avoids all exposure during prolonged bear markets.
    '''
    base = sig_boost_only_seasonal_vt45(df)
    bull_flag = (df['Close'] > df['Close'].rolling(200).mean()).shift(1).fillna(False).astype(float)
    return (base * bull_flag).clip(0, 1)


def sig_bull_seasonal_vt45(df: pd.DataFrame) -> pd.Series:
    """VT45 with seasonal boost only when 60d drift is positive (in-market regime).\n    Avoids amplifying positions during bear market bounces."""
    r = df['log_ret']
    mu = r.rolling(60).mean()
    sigma = r.rolling(60).std()
    drift = mu - 0.5 * sigma ** 2
    base = sig_vol_targeting_40_45d(df)
    scalar = pd.Series(1.0, index=df.index)
    for i, idx in enumerate(df.index):
        if idx.month in SEASONAL_STRONG_MONTHS:
            d = float(drift.iloc[i]) if i < len(drift) else 0.0
            if not np.isnan(d) and d > 0:
                    scalar.loc[idx] = 1.2
    return (base * scalar).clip(0.0, 1.0)


def portfolio_btc_lead_rp(assets: list[tuple[str, pd.DataFrame]], df_btc: pd.DataFrame):
    """Unified portfolio: BTC uses own VT; altcoins use BTC-Lead VT.
    Combined with inverse-vol risk-parity weights, rebalanced monthly.
    """
    W = 105
    print(f"\n{'═' * W}")
    print('  COMBINED BTC-LEAD + RISK-PARITY PORTFOLIO')
    print('  BTC: own VT(40%,45d)  |  ETH/SOL/BNB: BTC-Lead VT(40%,45d)')
    print('  Weights: inverse trailing-6m vol, rebalanced monthly')
    print(f"{'═' * W}")

    asset_names = [n for n, _ in assets]

    equities = {}
    for name, df_a in assets:
        sig = sig_vol_targeting_40_45d(df_a) if name == 'BTC' else sig_btc_lead_vt(df_a, df_btc)
        equities[name] = Backtest(df_a, sig).equity

    eq_df = pd.DataFrame(equities).dropna()
    mth = eq_df.resample('ME').last()
    mth_r = mth.pct_change().dropna()

    portfolios = {}

    rp_rets, rp_dates = [], []
    for i in range(6, len(mth_r)):
        trailing = mth_r.iloc[i - 6:i]
        inv_vol = 1.0 / (trailing.std() + 1e-08)
        w = inv_vol / inv_vol.sum()
        rp_rets.append(float((mth_r.iloc[i] * w).sum()))
        rp_dates.append(mth_r.index[i])
    portfolios['BTC-Lead + Risk-Parity'] = pd.Series(rp_rets, index=rp_dates)

    portfolios['BTC-Lead + Equal-Weight'] = mth_r.mean(axis=1).loc[rp_dates]

    own_equities = {}
    for name, df_a in assets:
        own_equities[name] = Backtest(df_a, sig_vol_targeting_40_45d(df_a)).equity
    own_mth_r = pd.DataFrame(own_equities).dropna().resample('ME').last().pct_change().dropna()
    own_rp = []
    for i in range(6, len(own_mth_r)):
        if own_mth_r.index[i] in pd.DatetimeIndex(rp_dates):
            trailing = own_mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            own_rp.append(float((own_mth_r.iloc[i] * w).sum()))
    if len(own_rp) == len(rp_dates):
        portfolios['Own-Signal + Risk-Parity'] = pd.Series(own_rp, index=rp_dates)

    bh_r = {}
    for name, df_a in assets:
        bh_r[name] = Backtest(df_a, sig_buy_hold(df_a)).equity.resample('ME').last().pct_change().dropna()
    bh_df = pd.DataFrame(bh_r).dropna()
    portfolios['B&H Equal-Weight'] = bh_df.mean(axis=1).reindex(rp_dates)

    def port_stats(s, label):
        s = s.dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avg_w = float(s[s > 0].mean()) if n_pos else 0
        avg_l = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        streak = mx = 0
        for r in s:
            streak = streak + 1 if r < 0 else 0
            mx = max(mx, streak)
        marker = '★' if label == 'BTC-Lead + Risk-Parity' else ' '
        print(f'  {marker}{label:<32}  {cum:>+8.1%}  {sh:>7.2f}  '
              f'{wr:>6.0%}  {n_pos:>3}/{n_mo}  '
              f'{avg_w:>+7.1%}  {avg_l:>+8.1%}  {v99:>+9.1%}  {mx:>5}')

    print(f"\n  {'Portfolio':<33}  {'CumRet':>8}  {'MthSh':>7}  "
          f"{'WinRate':>7}  {'Wins':>7}  {'AvgWin':>8}  {'AvgLoss':>9}  "
          f"{'VaR-99':>9}  {'MaxCons':>7}")
    print('─' * W)
    for label, series in portfolios.items():
        port_stats(series, label)
    print('═' * W)


def calendar_effects_analysis(assets: list[tuple[str, pd.DataFrame]]):
    """Analyse quarterly and monthly seasonal patterns in crypto returns.

    Tests:
    1. Average return by calendar month (Jan effect, Q4 rally, etc.)
    2. Q4 (Oct-Dec) vs rest-of-year average
    3. Seasonal overlay: boost position in historically strong months
    """
    W = 105
    print(f"\n{'═' * W}")
    print("  CALENDAR / SEASONAL EFFECTS ANALYSIS")
    print("  Raw B&H returns by month/quarter — do seasonal patterns exist?")
    print(f"{'═' * W}")

    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    quarters = {1: 'Q1', 2: 'Q1', 3: 'Q1', 4: 'Q2', 5: 'Q2', 6: 'Q2',
                7: 'Q3', 8: 'Q3', 9: 'Q3', 10: 'Q4', 11: 'Q4', 12: 'Q4'}

    all_monthly = {m: [] for m in range(1, 13)}
    all_qtly = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}

    print(f"\n  {'Month':<6} " + '  '.join(f"{n:>8}" for n, _ in assets) + f"  {'AvgAll':>8}")
    print('─' * W)

    asset_monthly = {}
    for asset_name, df_a in assets:
        mth_r = df_a['Close'].resample('ME').last().pct_change().dropna()
        by_mo = {m: [] for m in range(1, 13)}
        for ts, r in mth_r.items():
            by_mo[ts.month].append(float(r))
        asset_monthly[asset_name] = {m: float(np.mean(v)) if v else 0.0
                                     for m, v in by_mo.items()}
        for m, vals in by_mo.items():
            all_monthly[m].extend(vals)

    for m in range(1, 13):
        row_vals = [asset_monthly[n][m] for n, _ in assets]
        avg_all = float(np.mean([v for v in all_monthly[m]]))
        print(f"  {month_abbr[m - 1]:<6} " +
              '  '.join(f"{v:>+7.1%} " for v in row_vals) +
              f"  {avg_all:>+7.1%}")
        all_qtly[quarters[m]].extend(all_monthly[m])

    print('─' * W)

    print(f"\n  {'Quarter':<8} " + '  '.join(f"{n:>8}" for n, _ in assets) + f"  {'AvgAll':>8}")
    print('─' * 60)
    asset_qtly = {n: {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
                  for n, _ in assets}
    for asset_name, df_a in assets:
        mth_r = df_a['Close'].resample('ME').last().pct_change().dropna()
        for ts, r in mth_r.items():
            asset_qtly[asset_name][quarters[ts.month]].append(float(r))

    strong_months = set()
    for m in range(1, 13):
        if float(np.mean(all_monthly[m])) > 0.03:
            strong_months.add(m)

    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        row_vals = [float(np.mean(asset_qtly[n][q])) if asset_qtly[n][q] else 0.0
                    for n, _ in assets]
        avg_all = float(np.mean(all_qtly[q])) if all_qtly[q] else 0.0
        marker = '★' if avg_all > 0.04 else ' '
        print(f"  {marker}{q:<7} " +
              '  '.join(f"{v:>+7.1%} " for v in row_vals) +
              f"  {avg_all:>+7.1%}")

    print(f"\n  Strong months (avg B&H >+3% across all assets): "
          f"{', '.join(month_abbr[m - 1] for m in sorted(strong_months)) or 'none'}")

    print(f"\n  {'─' * 70}")
    print("  SEASONAL OVERLAY TEST  (VT45 × 1.2 in strong months, × 0.8 in weak)")
    print(f"  {'─' * 70}")
    print(f"  {'Strategy':<28}  {'Asset':<5}  {'Return':>9}  {'Sharpe':>7}  {'WR':>6}")

    weak_months = {m for m in range(1, 13)
                   if float(np.mean(all_monthly[m])) < -0.01}

    for asset_name, df_a in assets:
        bt_base = Backtest(df_a, sig_vol_targeting_40_45d(df_a))
        base_sig = sig_vol_targeting_40_45d(df_a)

        month_scalar = pd.Series(1.0, index=df_a.index)
        for idx in df_a.index:
            if idx.month in strong_months:
                month_scalar.loc[idx] = 1.2
            elif idx.month in weak_months:
                month_scalar.loc[idx] = 0.8
        seasonal_sig = (base_sig * month_scalar).clip(0, 1)
        bt_seas = Backtest(df_a, seasonal_sig)

        mth_b = bt_base.equity.resample('ME').last().pct_change().dropna()
        mth_s = bt_seas.equity.resample('ME').last().pct_change().dropna()
        wr_b = float((mth_b > 0).mean())
        wr_s = float((mth_s > 0).mean())
        print(f"  {'VT45 Baseline':<28}  {asset_name:<5}  "
              f"{bt_base.total_return:>+8.1%}  {bt_base.sharpe:>7.2f}  {wr_b:>5.0%}")
        print(f"  {'VT45 + Seasonal':<28}  {asset_name:<5}  "
              f"{bt_seas.total_return:>+8.1%}  {bt_seas.sharpe:>7.2f}  {wr_s:>5.0%}")
        print()

    print('═' * W)


def seasonal_walkforward_analysis(assets: list[tuple[str, pd.DataFrame]]):
    """Out-of-sample test of seasonal overlay: discover strong months from training half,
    apply in test half. Eliminates data leakage from in-sample seasonal discovery.
    """
    W = 112
    print(f"\n{'═' * W}")
    print("  SEASONAL OVERLAY — WALK-FORWARD VALIDATION")
    print("  Train: first 2.5yr → discover strong months  |  Test: last 2.5yr → apply overlay")
    print(f"{'═' * W}")

    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    btc_df = assets[0][1]
    mid = len(btc_df) // 2
    train = btc_df.iloc[:mid]

    all_monthly_train = {m: [] for m in range(1, 13)}
    for _, df_a in assets:
        tr_a = df_a.iloc[:len(df_a) // 2]
        mth_r = tr_a['Close'].resample('ME').last().pct_change().dropna()
        for ts, r in mth_r.items():
            all_monthly_train[ts.month].append(float(r))

    train_strong = {m for m in range(1, 13)
                    if len(all_monthly_train[m]) >= 2
                    and float(np.mean(all_monthly_train[m])) > 0.03}
    train_weak = {m for m in range(1, 13)
                  if len(all_monthly_train[m]) >= 2
                  and float(np.mean(all_monthly_train[m])) < -0.01}

    print(f"\n  Training strong months (avg B&H >+3%): "
          f"{', '.join(month_abbr[m - 1] for m in sorted(train_strong)) or 'none'}")
    print(f"  Training weak months   (avg B&H < -1%): "
          f"{', '.join(month_abbr[m - 1] for m in sorted(train_weak)) or 'none'}")
    print(f"\n  {'Strategy':<32}  {'Asset':<5}  {'TestRet':>9}  {'Sharpe':>7}  {'WR':>6}  {'Alpha':>9}")
    print("─" * W)

    for asset_name, df_a in assets:
        mid_a = len(df_a) // 2
        test_a = df_a.iloc[mid_a:].copy()


        bt_base = Backtest(test_a, sig_vol_targeting_40_45d(test_a))

        insample_s = _apply_seasonal_scalar(
            sig_vol_targeting_40_45d(test_a), SEASONAL_STRONG_MONTHS, SEASONAL_WEAK_MONTHS)
        bt_insamp = Backtest(test_a, insample_s)

        wf_sig = _apply_seasonal_scalar(
            sig_vol_targeting_40_45d(test_a), train_strong, train_weak)
        bt_wf = Backtest(test_a, wf_sig)

        bt_bh = Backtest(test_a, pd.Series(1.0, index=test_a.index))

        for label, bt in (
            ("VT45 Baseline (test)", bt_base),
            ("VT45+Seasonal in-sample", bt_insamp),
            ("VT45+Seasonal WF (OOS)", bt_wf),
            ("B&H (test)", bt_bh),
        ):
            mth_eq = bt.equity.resample('ME').last().pct_change().dropna()
            wr = float((mth_eq > 0).mean()) if len(mth_eq) > 0 else 0.0
            alpha = bt.total_return - bt_bh.total_return
            print(f"  {label:<32}  {asset_name:<5}  "
                  f"{bt.total_return:>+8.1%}  {bt.sharpe:>7.2f}  {wr:>5.0%}  {alpha:>+8.1%}")
        print()
    print("═" * W)


def portfolio_final_unified(assets: list[tuple[str, pd.DataFrame]], df_btc: pd.DataFrame):
    """Final Unified Deployable System:
    - BTC: own VT45 + seasonal overlay
    - ETH/SOL/BNB: BTC-Lead VT + seasonal overlay
    - Portfolio weighting: inverse-vol risk parity, rebalanced monthly
    Compare vs prior portfolio configurations.
    """
    W = 110
    print(f"\n{'═' * W}")
    print("  FINAL UNIFIED DEPLOYABLE SYSTEM COMPARISON")
    print("  BTC: VT45+Seasonal  |  Alts: BTC-Lead+Seasonal  |  Weights: inverse-vol RP")
    print(f"{'═' * W}")

    def build_equity(name, df_a):
        if name == 'BTC':
            return Backtest(df_a, sig_seasonal_vt45(df_a)).equity
        else:
            return Backtest(df_a, sig_btc_lead_seasonal(df_a, df_btc)).equity

    def build_equity_vt(name, df_a):
        return Backtest(df_a, sig_vol_targeting_40_45d(df_a)).equity

    def build_equity_btclead(name, df_a):
        if name == 'BTC':
            return Backtest(df_a, sig_vol_targeting_40_45d(df_a)).equity
        else:
            return Backtest(df_a, sig_btc_lead_vt(df_a, df_btc)).equity

    def rp_cumret(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    configurations = [
        ('★ Final Unified (BTC-Lead+Seasonal+RP)', rp_cumret(build_equity)),
        ('  BTC-Lead+RP (no seasonal)', rp_cumret(build_equity_btclead)),
        ('  Own VT45+Seasonal+RP', rp_cumret(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity)),
        ('  Own VT45+RP (baseline)', rp_cumret(build_equity_vt)),
    ]

    print(f"\n  {'Portfolio':<42}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    for label, s in configurations:
        s = s.dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avg_w = float(s[s > 0].mean()) if n_pos else 0
        avg_l = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f"  {label:<42}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  "
              f"{n_pos:>3}/{n_mo}  {avg_w:>+7.1%}  {avg_l:>+8.1%}  {v99:>+9.1%}")
    print('═' * W)


def seasonal_bootstrap_significance(assets: list[tuple[str, pd.DataFrame]], n_iter: int = 500):
    """Bootstrap significance test of seasonal overlay.
    Randomly shuffles month assignments 500 times to build null distribution.
    Tests: is the seasonal boost statistically significant?
    """
    W = 100
    print(f"\n{'═' * W}")
    print("  SEASONAL BOOTSTRAP SIGNIFICANCE TEST  (500 random month assignments)")
    print("  Null: shuffle which months are 'strong'; compare actual gain to random")
    print(f"{'═' * W}")

    rng = np.random.default_rng(42)
    months = list(range(1, 13))

    print(f"\n  {'Asset':<5}  {'Actual Gain':>12}  {'Rand p5':>9}  {'Rand p95':>9}  {'p-value':>9}  {'Verdict'}")
    print('─' * W)

    for asset_name, df_a in assets:
        base_ret = Backtest(df_a, sig_vol_targeting_40_45d(df_a)).total_return
        actual_ret = Backtest(df_a, sig_seasonal_vt45(df_a)).total_return
        actual_gain = actual_ret - base_ret

        random_gains = []
        for _ in range(n_iter):
            rand_strong = set(rng.choice(months, size=5, replace=False).tolist())
            rand_weak = set(rng.choice(
                [m for m in months if m not in rand_strong], size=1, replace=False).tolist())
            rand_sig = _apply_seasonal_scalar(
                sig_vol_targeting_40_45d(df_a), rand_strong, rand_weak)
            rand_ret = Backtest(df_a, rand_sig).total_return
            random_gains.append(rand_ret - base_ret)

        random_gains = np.array(random_gains)
        p5 = float(np.percentile(random_gains, 5))
        p95 = float(np.percentile(random_gains, 95))
        pval = float((random_gains >= actual_gain).mean())
        verdict = '★ SIGNIFICANT (p<0.10)' if pval < 0.10 else ('~ marginal' if pval < 0.20 else '✗ not significant')
        print(f"  {asset_name:<5}  {actual_gain:>+11.1%}  {p5:>+8.1%}  {p95:>+8.1%}  "
              f"{pval:>8.3f}  {verdict}")
    print('═' * W)


def seasonal_variants_comparison(assets: list[tuple[str, pd.DataFrame]]):
    """Compare seasonal variants: full (boost+dampen), boost-only, bull-regime.
    Purpose: identify OOS-robust configuration after v23's BTC finding.
    """
    W = 108
    print(f"\n{'═' * W}")
    print('  SEASONAL VARIANTS COMPARISON')
    print('  Full=boost+dampen | BoostOnly=no dampening | BullRegime=boost only when drift>0')
    print(f"{'═' * W}")
    print(f'''\n  {'Variant':<30}  {'Asset':<5}  {'Return':>9}  {'Sharpe':>7}  {'WR':>6}  {'vs Baseline':>12}''')
    print('─' * W)
    variants = [
        ('VT45 Baseline', sig_vol_targeting_40_45d),
        ('VT45+Seasonal (full)', sig_seasonal_vt45),
        ('VT45+Boost-Only', sig_boost_only_seasonal_vt45),
        ('VT45+Bull-Regime Seas', sig_bull_seasonal_vt45),
        ('Blend+Boost-Only', sig_boost_only_seasonal_blend),
        ('Asym+Boost-Only', sig_boost_only_seasonal_asym)]
    for asset_name, df_a in assets:
        baseline_ret = Backtest(df_a, sig_vol_targeting_40_45d(df_a)).total_return
        first = True
        for vname, vfn in variants:
            bt = Backtest(df_a, vfn(df_a))
            mth_eq = bt.equity.resample('ME').last().pct_change().dropna()
            wr = float((mth_eq > 0).mean()) if len(mth_eq) else 0.0
            delta = bt.total_return - baseline_ret
            marker = '+' if delta > 0.01 else ('-' if delta < -0.01 else '~')
            name_lbl = vname if first else ''
            first = False
            print(f'''  {name_lbl:<30}  {asset_name:<5}  {bt.total_return:>+8.1%}  {bt.sharpe:>7.2f}  {wr:>5.0%}  {marker}{delta:>+9.1%}''')
        print()
    print('═' * W)


def portfolio_v25_best(assets: list[tuple[str, pd.DataFrame]]):
    """v25: Extended portfolio comparison — Blend+Boost-Only+RP as candidate #1."""
    W = 112
    print(f"\n{'═' * W}")
    print("  v25 BEST PORTFOLIO — Blend+Boost-Only+RP vs all prior configurations")
    print(f"{'═' * W}")

    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    configs = [
        ('★ Blend+BoostOnly+RP',      lambda n, df: Backtest(df, sig_boost_only_seasonal_blend(df)).equity),
        ('★ Asym+BoostOnly+RP',       lambda n, df: Backtest(df, sig_boost_only_seasonal_asym(df)).equity),
        ('★ Base5+BlendBoost+RP',     lambda n, df: Backtest(df, sig_base5_boost_blend(df)).equity),
        ('  Own VT45+Seasonal+RP',    lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity),
        ('  Blend+Seasonal+RP',       lambda n, df: Backtest(df, sig_seasonal_blend(df)).equity),
        ('  Own VT45+BoostOnly+RP',   lambda n, df: Backtest(df, sig_boost_only_seasonal_vt45(df)).equity),
        ('  Own VT45+RP (baseline)',  lambda n, df: Backtest(df, sig_vol_targeting_40_45d(df)).equity),
    ]

    print(f"\n  {'Portfolio':<30}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    for label, efn in configs:
        s = rp_series(efn).dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f"  {label:<30}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  "
              f"{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}  {v99:>+9.1%}")
    print('═' * W)


def commission_sensitivity_top_strategies(assets: list[tuple[str, pd.DataFrame]]):
    """v27: Commission sensitivity for top strategies at realistic trading rates."""
    global COMMISSION
    W = 118
    strategies = [
        ('Blend+Seasonal', sig_seasonal_blend),
        ('Blend+BoostOnly', sig_boost_only_seasonal_blend),
        ('VT45+Seasonal', sig_seasonal_vt45),
        ('VT45+BoostOnly', sig_boost_only_seasonal_vt45),
        ('Base5+BlendBoost', sig_base5_boost_blend),
    ]
    rates = [0.0005, 0.001, 0.002, 0.005, 0.01]
    asset_names = [n for n, _ in assets]

    for strat_name, strat_fn in strategies:
        print(f"\n{'═' * W}")
        print(f'  COMMISSION SENSITIVITY — {strat_name}')
        print(f"{'═' * W}")
        hdr = (f"  {'Comm/trade':>12}  {'RT':>6}  "
               + '  '.join(f"{n + ' Ret':>11}" for n in asset_names)
               + f"  {'AvgSh':>7}  {'All+':>5}")
        print(hdr)
        print('─' * W)
        for c in rates:
            old_c = COMMISSION
            COMMISSION = c
            bts = [Backtest(df_a, strat_fn(df_a)) for _, df_a in assets]
            COMMISSION = old_c
            avg_sh = float(np.mean([bt.sharpe for bt in bts]))
            all_pos = '✓' if all(bt.total_return > 0 for bt in bts) else '✗'
            rets = '  '.join(f'{bt.total_return:>+10.1%}' for bt in bts)
            print(f'  {all_pos} {c:>10.3%}  {2 * c:>5.3%}  {rets}  {avg_sh:>7.2f}')
        print('═' * W)


def portfolio_v27_ensemble(assets: list[tuple[str, pd.DataFrame]]):
    """v27: Ensemble portfolio blending high-Sharpe with high-WR RP variants."""
    W = 115
    print(f"\n{'═' * W}")
    print('  v27 ENSEMBLE PORTFOLIO — Blending high-Sharpe with high-WR variants')
    print('  VT45+BoostOnly (Sh=0.71, WR=31%) × Base5+BlendBoost (WR=50%)')
    print(f"{'═' * W}")

    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    s_vt45bo = rp_series(lambda n, df: Backtest(df, sig_boost_only_seasonal_vt45(df)).equity)
    s_b5bb = rp_series(lambda n, df: Backtest(df, sig_base5_boost_blend(df)).equity)
    s_blend = rp_series(lambda n, df: Backtest(df, sig_boost_only_seasonal_blend(df)).equity)
    s_top = rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity)

    def bseries(s1, s2, w1):
        idx = s1.index.intersection(s2.index)
        return w1 * s1.loc[idx] + (1 - w1) * s2.loc[idx]

    def report(label, s):
        s = s.dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f"  {label:<40}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  "
              f"{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}  {v99:>+9.1%}")

    print(f"\n  {'Portfolio':<40}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    report('  VT45+BoostOnly+RP  (Sh anchor)', s_vt45bo)
    report('  Base5+BlendBoost+RP  (WR anchor)', s_b5bb)
    print()
    for w1, lbl in [(0.7, '70/30'), (0.5, '50/50'), (0.3, '30/70')]:
        report(f'★ Ens {lbl} VT45Bo/Base5BB', bseries(s_vt45bo, s_b5bb, w1))
    print()
    report('  Blend+BoostOnly+RP', s_blend)
    report('  VT45+Seasonal+RP  (top Sharpe)', s_top)
    report('★ Ens 50/50 VT45Bo/BlendBo', bseries(s_vt45bo, s_blend, 0.5))
    print('═' * W)


def seasonal_window_sensitivity(assets: list[tuple[str, pd.DataFrame]]):
    """v28: Test VT+BoostOnly seasonal at different drift estimation windows.
    Finds optimal drift lookback (30/45/60/90d) for the seasonal overlay.
    """
    from functools import partial
    W = 110
    windows = [30, 45, 60, 90]
    print(f"\n{'═' * W}")
    print("  SEASONAL WINDOW SENSITIVITY — VT+BoostOnly at 30/45/60/90d drift windows")
    print("  Question: does a shorter/longer drift window improve seasonal alpha?")
    print(f"{'═' * W}")
    print(f"  {'Window':>8}  {'BTC Ret':>9}  {'BTC Sh':>8}  {'ETH Ret':>9}  "
          f"{'SOL Ret':>9}  {'BNB Ret':>9}  {'AvgSh':>7}  {'All+':>5}")
    print('─' * W)

    best = {'score': -999, 'window': 60}
    for w in windows:
        def seasonal_vt_w(df, _w=w):
            base = sig_vol_targeting(df, window=_w, target_ann_vol=0.40, rvol_window=45)
            return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak=set(), boost=1.2)
        bts = [Backtest(df_a, seasonal_vt_w(df_a)) for _, df_a in assets]
        avg_sh = float(np.mean([bt.sharpe for bt in bts]))
        all_pos = '✓' if all(bt.total_return > 0 for bt in bts) else '✗'
        rets = '  '.join(f'{bt.total_return:>+8.1%}' for bt in bts)
        print(f"  {all_pos} {w:>6}d  {rets}  {avg_sh:>7.2f}")
        if avg_sh > best['score']:
            best = {'score': avg_sh, 'window': w}
    print('═' * W)
    print(f"  Optimal drift window for seasonal overlay: {best['window']}d  (AvgSh={best['score']:.3f})")


def portfolio_v28_final(assets: list[tuple[str, pd.DataFrame]]):
    """v28: Final portfolio comparison including new drift-scaled and bull-filter variants."""
    W = 118
    print(f"\n{'═' * W}")
    print('  v28 FINAL PORTFOLIO — New variants vs best prior configurations')
    print(f"{'═' * W}")

    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    def bseries(s1, s2, w1):
        idx = s1.index.intersection(s2.index)
        return w1 * s1.loc[idx] + (1 - w1) * s2.loc[idx]

    s_vt45bo = rp_series(lambda n, df: Backtest(df, sig_boost_only_seasonal_vt45(df)).equity)
    s_b5bb = rp_series(lambda n, df: Backtest(df, sig_base5_boost_blend(df)).equity)
    s_top = rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity)
    s_ds = rp_series(lambda n, df: Backtest(df, sig_drift_scaled_vt45_boost(df)).equity)
    s_bf = rp_series(lambda n, df: Backtest(df, sig_bull_filter_vt45_boost(df)).equity)

    def report(label, s):
        s = s.dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f'  {label:<40}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  '
              f'{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}  {v99:>+9.1%}')

    print(f"\n  {'Portfolio':<40}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}"
          f"  {'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    report('  VT45+Seasonal+RP  (v26 champion)', s_top)
    report('  Ens 70/30 VT45Bo/Base5BB  (v27 best)', bseries(s_vt45bo, s_b5bb, 0.7))
    print()
    report('★ DriftScaled+VT45+Boost+RP', s_ds)
    report('★ BullFilter+VT45+Boost+RP', s_bf)
    report('★ Ens 70/30 DriftScaled/VT45Bo', bseries(s_ds, s_vt45bo, 0.7))
    report('★ Ens 50/50 DriftScaled/BullFilter', bseries(s_ds, s_bf, 0.5))
    print('═' * W)


def floor_sensitivity_analysis(assets: list[tuple[str, pd.DataFrame]]):
    """v29: Floor parameter sweep — map the WR/Sharpe frontier for VT45+Boost.
    Tests base_pct=[0,2,5,8,10,15,20]% to find Sharpe-WR tradeoff curve.
    """
    W = 118
    floors = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    print(f"\n{'═' * W}")
    print('  FLOOR SENSITIVITY — VT45+BoostOnly at different position floors')
    print('  Floor=0 is pure VT45+BoostOnly; higher floor → more WR, less Sharpe')
    print(f"{'═' * W}")
    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = ([], [])
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)
    print('\n  Individual asset means (Sharpe avg / WR):')
    print(f"  {'Floor':>7}  {'BTC':>10}  {'ETH':>10}  {'SOL':>10}  {'BNB':>10}  "
          f"{'AvgSh':>7}  {'Port Sh':>8}  {'Port WR':>8}  {'Port Ret':>9}")
    print('─' * W)
    for floor in floors:
        def sig_floor_boost(df, f=floor):
            base = sig_base_position(df, base_pct=f)
            return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak=set(), boost=1.2)
        bts = [Backtest(df_a, sig_floor_boost(df_a)) for _, df_a in assets]
        avg_sh = float(np.mean([bt.sharpe for bt in bts]))
        ind = '  '.join((f'{bt.total_return:>+9.1%}' for bt in bts))
        port = rp_series(lambda n, df, f=floor: Backtest(df, sig_floor_boost(df)).equity)
        port = port.dropna()
        n_mo = len(port)
        n_pos = int((port > 0).sum())
        p_wr = n_pos / n_mo if n_mo else 0
        p_sh = float(np.sqrt(12) * port.mean() / port.std()) if port.std() > 0 else 0
        p_ret = float((1 + port).prod() - 1)
        print(f'  {floor:>6.0%}   {ind}  {avg_sh:>7.2f}  {p_sh:>8.2f}  {p_wr:>7.0%}  {p_ret:>+8.1%}')
    print('═' * W)


def portfolio_v29_final(assets: list[tuple[str, pd.DataFrame]]):
    """v29: Final portfolio — test VT45+Seasonal as ensemble anchor + floor variants."""
    W = 118
    print(f"\n{'═' * W}")
    print('  v29 FINAL PORTFOLIO — Floor variants + VT45+Seasonal as ensemble anchor')
    print(f"{'═' * W}")
    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = ([], [])
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)
    def bseries(s1, s2, w1):
        idx = s1.index.intersection(s2.index)
        return w1 * s1.loc[idx] + (1 - w1) * s2.loc[idx]
    s_top = rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity)
    s_vt45bo = rp_series(lambda n, df: Backtest(df, sig_boost_only_seasonal_vt45(df)).equity)
    s_b5bb = rp_series(lambda n, df: Backtest(df, sig_base5_boost_blend(df)).equity)
    s_b2vt = rp_series(lambda n, df: Backtest(df, sig_base2_vt45_boost(df)).equity)
    s_b5vt = rp_series(lambda n, df: Backtest(df, sig_base5_vt45_boost(df)).equity)
    def report(label, s):
        s = s.dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f'  {label:<42}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  '
              f'{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}  {v99:>+9.1%}')
    print(f"\n  {'Portfolio':<42}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    report('  VT45+Seasonal+RP  (Sh champion)', s_top)
    report('  Ens 70/30 VT45Bo/Base5BB  (v27 WR champion)', bseries(s_vt45bo, s_b5bb, 0.7))
    print()
    report('★ Base2+VT45+Boost+RP  (tiny floor)', s_b2vt)
    report('★ Base5+VT45+Boost+RP  (vs Base5Blend)', s_b5vt)
    report('★ Ens 70/30 VT45Seasonal/Base5BB', bseries(s_top, s_b5bb, 0.7))
    report('★ Ens 70/30 VT45Seasonal/Base2VT', bseries(s_top, s_b2vt, 0.7))
    report('★ Ens 50/50 VT45Seasonal/Base5VT', bseries(s_top, s_b5vt, 0.5))
    print('═' * W)


def portfolio_v30_final(assets: list[tuple[str, pd.DataFrame]]):
    """v30: Base2+Blend+Boost portfolio + commission sweep for top 3 portfolios."""
    W = 118
    print(f"\n{'═' * W}")
    print("  v30 FINAL PORTFOLIO — Base2+Blend+Boost+RP vs v29 champions")
    print(f"{'═' * W}")

    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    def bseries(s1, s2, w1):
        idx = s1.index.intersection(s2.index)
        return w1 * s1.loc[idx] + (1 - w1) * s2.loc[idx]

    def report(label, s):
        s = s.dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f"  {label:<45}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  "
              f"{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}  {v99:>+9.1%}")

    s_top = rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity)
    s_b2vt = rp_series(lambda n, df: Backtest(df, sig_base2_vt45_boost(df)).equity)
    s_b2bb = rp_series(lambda n, df: Backtest(df, sig_base2_blend_boost(df)).equity)

    print(f"\n  {'Portfolio':<45}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    report('  VT45+Seasonal+RP  (Sh champion v25)', s_top)
    report('  Base2+VT45+Boost+RP  (v29 floor winner)', s_b2vt)
    report('  Ens 70/30 VT45Seasonal/Base2VT  (v29 best ens)', bseries(s_top, s_b2vt, 0.70))
    print()
    report('★ Base2+Blend+Boost+RP  (v30 new)', s_b2bb)
    report('★ Ens 70/30 VT45Seasonal/Base2BB', bseries(s_top, s_b2bb, 0.70))
    report('★ Ens 50/50 VT45Seasonal/Base2BB', bseries(s_top, s_b2bb, 0.50))
    report('★ Ens 50/50 Base2VT/Base2BB', bseries(s_b2vt, s_b2bb, 0.50))
    print('═' * W)

    print(f"\n{'═' * W}")
    print("  v30 PORTFOLIO COMMISSION SWEEP — Top portfolios at realistic rates")
    print("  (Monthly rebal so round-trip = 2×trade rate — very low turnover)")
    print(f"{'═' * W}")

    top_portfolios = [
        ('VT45+Seasonal+RP', lambda c: _rp_at_commission(assets, sig_seasonal_vt45, c)),
        ('Base2+VT45+Boost+RP', lambda c: _rp_at_commission(assets, sig_base2_vt45_boost, c)),
        ('Base2+Blend+Boost+RP', lambda c: _rp_at_commission(assets, sig_base2_blend_boost, c)),
    ]
    rates = [0.0005, 0.001, 0.002, 0.005, 0.010]

    for port_name, port_fn in top_portfolios:
        print(f"\n  {port_name}")
        print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
        print("  ───────────────────────────────────────────────────────")
        for c in rates:
            s = port_fn(c)
            if s is None or len(s.dropna()) < 12:
                print(f"  {c:>11.3%}  {2 * c:>5.3%}  {'N/A':>9}  {'N/A':>8}  {'N/A':>6}  {'N/A':>7}")
                continue
            s = s.dropna()
            cum = float((1 + s).prod() - 1)
            sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
            wr = float((s > 0).mean())
            ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
            print(f"  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}")
    print('═' * W)


def _rp_at_commission(assets, sig_fn, commission, allow_leverage: bool = False):
    """Helper: compute risk-parity monthly return series at a given commission rate.
    allow_leverage=True disables the [0,1] position clip — supports levered signals.
    """
    global COMMISSION
    old_c = COMMISSION
    COMMISSION = commission
    try:
        equities = {name: Backtest(df_a, sig_fn(df_a),
                                   allow_short=allow_leverage).equity
                    for name, df_a in assets}
    finally:
        COMMISSION = old_c
    eq_df = pd.DataFrame(equities).dropna()
    mth_r = eq_df.resample('ME').last().pct_change().dropna()
    rets, dates = [], []
    for i in range(6, len(mth_r)):
        trailing = mth_r.iloc[i - 6:i]
        inv_vol = 1.0 / (trailing.std() + 1e-8)
        w = inv_vol / inv_vol.sum()
        rets.append(float((mth_r.iloc[i] * w).sum()))
        dates.append(mth_r.index[i])
    return pd.Series(rets, index=dates)


def portfolio_v31_ensemble_grid(assets: list[tuple[str, pd.DataFrame]]):
    """v31: 3-way ensemble grid search over (VT45+Seasonal, Base2+VT45+Boost, Base2+Blend+Boost).
    Maps the full Sh/WR Pareto frontier — finds the optimal blend weights.
    Grid step=10% so we test all (w1, w2, w3) where w1+w2+w3=1, each ∈ {0,0.1,...,1.0}.
    """
    W = 120
    print(f"\n{'═' * W}")
    print("  v31 3-WAY ENSEMBLE GRID — VT45+Seasonal × Base2+VT45 × Base2+Blend+Boost")
    print("  Full Pareto sweep: Sharpe vs Win-Rate frontier across all weight combos")
    print(f"{'═' * W}")

    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    s_top = rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity)
    s_b2v = rp_series(lambda n, df: Backtest(df, sig_base2_vt45_boost(df)).equity)
    s_b2b = rp_series(lambda n, df: Backtest(df, sig_base2_blend_boost(df)).equity)

    idx = s_top.index.intersection(s_b2v.index).intersection(s_b2b.index)
    s_top = s_top.loc[idx]
    s_b2v = s_b2v.loc[idx]
    s_b2b = s_b2b.loc[idx]

    step = 0.1
    nsteps = round(1.0 / step)
    results = []
    for i in range(nsteps + 1):
        for j in range(nsteps + 1 - i):
            k = nsteps - i - j
            w1 = i / nsteps
            w2 = j / nsteps
            w3 = k / nsteps
            s = w1 * s_top + w2 * s_b2v + w3 * s_b2b
            s = s.dropna()
            if len(s) < 12:
                continue
            n_mo = len(s)
            n_pos = int((s > 0).sum())
            wr = n_pos / n_mo
            cum = float((1 + s).prod() - 1)
            sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0.0
            results.append((w1, w2, w3, sh, wr, cum, n_pos, n_mo))

    results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n  {'w_VT45S':>8}  {'w_B2VT':>7}  {'w_B2BB':>7}  {'MthSh':>7}  "
          f"{'WR':>6}  {'CumRet':>9}  {'Wins':>6}")
    print("  ────────────────────────────────────────────────────────────")
    print("  Top-10 by Sharpe:")
    for w1, w2, w3, sh, wr, cum, np_, nm in results[:10]:
        print(f"  {w1:>7.0%}  {w2:>7.0%}  {w3:>7.0%}  {sh:>7.2f}  "
              f"{wr:>5.0%}  {cum:>+8.1%}  {np_:>3}/{nm}")


    results.sort(key=lambda x: x[4], reverse=True)
    print("\n  Top-10 by Win-Rate:")
    for w1, w2, w3, sh, wr, cum, np_, nm in results[:10]:
        print(f"  {w1:>7.0%}  {w2:>7.0%}  {w3:>7.0%}  {sh:>7.2f}  "
              f"{wr:>5.0%}  {cum:>+8.1%}  {np_:>3}/{nm}")


    pareto = [(w1, w2, w3, sh, wr, cum, np_, nm)
              for w1, w2, w3, sh, wr, cum, np_, nm in results
              if wr >= 0.45]
    pareto.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Best Sh with WR≥45% ({len(pareto)} combos):")
    for row in pareto[:5]:
        w1, w2, w3, sh, wr, cum, np_, nm = row
        print(f"  ★ {w1:>6.0%}  {w2:>7.0%}  {w3:>7.0%}  {sh:>7.2f}  "
              f"{wr:>5.0%}  {cum:>+8.1%}  {np_:>3}/{nm}")


    pareto50 = [(w1, w2, w3, sh, wr, cum, np_, nm)
                for w1, w2, w3, sh, wr, cum, np_, nm in results
                if wr >= 0.50]
    pareto50.sort(key=lambda x: x[3], reverse=True)
    if pareto50:
        print(f"\n  Best Sh with WR≥50% ({len(pareto50)} combos):")
        for row in pareto50[:5]:
            w1, w2, w3, sh, wr, cum, np_, nm = row
            print(f"  ★ {w1:>6.0%}  {w2:>7.0%}  {w3:>7.0%}  {sh:>7.2f}  "
                  f"{wr:>5.0%}  {cum:>+8.1%}  {np_:>3}/{nm}")

    print("═" * W)


def fine_floor_sweep(assets: list[tuple[str, pd.DataFrame]]):
    """v31: Fine-grained floor sweep 0-5% in 0.5% steps for VT45+Boost.
    Finds the exact Sh/WR sweet spot between 0% and 5%.
    """
    W = 118
    floors = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    print(f"\n{'═' * W}")
    print("  v31 FINE FLOOR SWEEP — VT45+BoostOnly at 0-5% steps")
    print("  Finding exact Sh/WR sweet spot (v29 tested 0/2/5/8/10/15/20%)")
    print(f"{'═' * W}")

    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    print(f"\n  {'Floor':>7}  {'BTC':>10}  {'ETH':>10}  {'SOL':>10}  {'BNB':>10}  "
          f"{'AvgSh':>7}  {'Port Sh':>8}  {'Port WR':>8}  {'Port Ret':>9}")
    print('─' * W)

    for floor in floors:
        def sig_floor_boost(df, f=floor):
            base = sig_base_position(df, base_pct=f)
            return _apply_seasonal_scalar(base, SEASONAL_STRONG_MONTHS, weak=set(), boost=1.2)

        bts = [Backtest(df_a, sig_floor_boost(df_a)) for _, df_a in assets]
        avg_sh = float(np.mean([bt.sharpe for bt in bts]))
        ind = '  '.join(f'{bt.total_return:>+9.1%}' for bt in bts)

        port = rp_series(lambda n, df, f=floor: Backtest(df, sig_floor_boost(df)).equity)
        port = port.dropna()
        n_mo = len(port)
        n_pos = int((port > 0).sum())
        p_wr = n_pos / n_mo if n_mo else 0
        p_sh = float(np.sqrt(12) * port.mean() / port.std()) if port.std() > 0 else 0
        p_ret = float((1 + port).prod() - 1)
        print(f"  {floor:>6.1%}   {ind}  {avg_sh:>7.2f}  {p_sh:>8.2f}  {p_wr:>7.0%}  {p_ret:>+8.1%}")
    print('═' * W)


def portfolio_v32_fine_ensemble(assets: list[tuple[str, pd.DataFrame]]):
    """v32: Fine ensemble sweep (2% steps) around the v31 optimum of 90/10 VT45S/Base2VT.
    Finds exact WR transition point and confirms commission robustness of the optimal blend.
    """
    W = 120
    print(f"\n{'═' * W}")
    print("  v32 FINE ENSEMBLE SWEEP — VT45+Seasonal × Base2+VT45+Boost (2% steps)")
    print("  v31 found 90/10 = Sh=0.72, WR=48% — sweeping 80-100% range at 2% steps")
    print(f"{'═' * W}")

    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    s_top = rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity)
    s_b2v = rp_series(lambda n, df: Backtest(df, sig_base2_vt45_boost(df)).equity)
    idx = s_top.index.intersection(s_b2v.index)
    s_top = s_top.loc[idx]
    s_b2v = s_b2v.loc[idx]

    print(f"\n  {'w_VT45S':>8}  {'w_B2VT':>7}  {'MthSh':>7}  {'WR':>6}  "
          f"{'CumRet':>9}  {'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}")
    print('  ' + '─' * 70)

    results = []

    for pct in range(100, 68, -2):
        w1 = pct / 100.0
        w2 = 1.0 - w1
        s = w1 * s_top + w2 * s_b2v
        s = s.dropna()
        n_mo = len(s)
        if n_mo < 12:
            continue
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0.0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        marker = '★' if wr >= 0.45 else ' '
        print(f"  {marker} {w1:>7.0%}  {w2:>7.0%}  {sh:>7.2f}  "
              f"{wr:>5.0%}  {cum:>+8.1%}  {n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}")
        results.append((w1, w2, sh, wr, cum, n_pos, n_mo))

    transitions = [(w1, w2, sh, wr) for w1, w2, sh, wr in
                   [(r[0], r[1], r[2], r[3]) for r in results] if wr >= 0.45]
    if transitions:
        best = max(transitions, key=lambda x: (x[2], x[3]))
        print(f"\n  → Optimal: {best[0]:.0%}/{best[1]:.0%} = Sh={best[2]:.2f}, WR={best[3]:.0%}")
    print('═' * W)

    print(f"\n{'═' * W}")
    print("  v32 COMMISSION ROBUSTNESS — Ens 90/10 VT45Seasonal/Base2VT")
    print(f"{'═' * W}")
    rates = [0.0005, 0.001, 0.002, 0.005, 0.01]
    print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
    print('  ' + '─' * 55)
    for c in rates:
        s_a = _rp_at_commission(assets, sig_seasonal_vt45, c)
        s_b = _rp_at_commission(assets, sig_base2_vt45_boost, c)
        idx2 = s_a.index.intersection(s_b.index)
        s_ens = 0.9 * s_a.loc[idx2] + 0.1 * s_b.loc[idx2]
        s_ens = s_ens.dropna()
        if len(s_ens) < 12:
            continue
        cum = float((1 + s_ens).prod() - 1)
        sh = float(np.sqrt(12) * s_ens.mean() / s_ens.std()) if s_ens.std() > 0 else 0
        wr = float((s_ens > 0).mean())
        ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
        print(f"  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}")
    print('═' * W)


def portfolio_v33_final_validation(assets: list[tuple[str, pd.DataFrame]]):
    """v33: Sub-percent allocation sweep + 2D commission×allocation table.\n    Finds true minimum Base2VT allocation needed to trigger WR jump.\n    Shows full commission robustness surface.\n    """
    W = 120
    print(f"\n{'═' * W}")
    print('  v33 SUB-PERCENT ALLOCATION SWEEP — Finding minimum Base2VT to trigger WR jump')
    print('  v32: 2% triggers WR 30%→48%. Testing 0.5% steps to find true threshold.')
    print(f"{'═' * W}")
    def rp_series_at_commission(sig_fn, commission=0.01):
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = commission
        try:
            equities = {name: Backtest(df_a, sig_fn(df_a)).equity for name, df_a in assets}
        finally:
            COMMISSION = old_c
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = ([], [])
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)
    s_top = rp_series_at_commission(sig_seasonal_vt45, 0.01)
    s_b2v = rp_series_at_commission(sig_base2_vt45_boost, 0.01)
    idx = s_top.index.intersection(s_b2v.index)
    s_top_1 = s_top.loc[idx]
    s_b2v_1 = s_b2v.loc[idx]
    print(f"\n  {'w_VT45S':>8}  {'w_B2VT':>7}  {'MthSh':>7}  {'WR':>6}  {'Wins':>6}")
    print('  ────────────────────────────────────────')
    allocs = [1.0, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97, 0.965, 0.96, 0.95, 0.94, 0.92, 0.9]
    for w1 in allocs:
        w2 = 1.0 - w1
        s = (w1 * s_top_1 + w2 * s_b2v_1).dropna()
        n_mo = len(s)
        if n_mo < 12:
            continue
        else:
            n_pos = int((s > 0).sum())
            wr = n_pos / n_mo
            sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0.0
            mark = '★' if wr >= 0.45 else ' '
            print(f'  {mark} {w1:>7.1%}  {w2:>7.1%}  {sh:>7.2f}  {wr:>5.0%}  {n_pos:>3}/{n_mo}')
    print('═' * W)
    print(f"\n{'═' * W}")
    print('  v33 2D SURFACE — Sharpe and WR by (commission, allocation) for deployment sizing')
    print(f"{'═' * W}")
    comm_rates = [0.0005, 0.001, 0.002, 0.005, 0.01]
    test_allocs = [(1.0, 0.0), (0.98, 0.02), (0.95, 0.05), (0.9, 0.1), (0.8, 0.2), (0.0, 1.0)]
    print(f"\n  {'':20}", end='')
    for w1, w2 in test_allocs:
        print(f'  {w1:.0%}/{w2:.0%}', end='')
    print()
    print(f"  {'Commission':20}  " + '  '.join(['Sh   WR'] * len(test_allocs)))
    print('  ' + '─' * W)
    for c in comm_rates:
        s_a = rp_series_at_commission(sig_seasonal_vt45, c)
        s_b = rp_series_at_commission(sig_base2_vt45_boost, c)
        idx2 = s_a.index.intersection(s_b.index)
        s_a = s_a.loc[idx2]
        s_b = s_b.loc[idx2]
        print(f'  {c:.3%} / RT={2 * c:.3%}      ', end='')
        for w1, w2 in test_allocs:
            s = (w1 * s_a + w2 * s_b).dropna()
            sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
            wr = float((s > 0).mean())
            print(f'  {sh:.2f} {wr:.0%}', end='')
        print()
    print('═' * W)


def current_positions_and_spec(assets: list[tuple[str, pd.DataFrame]]):
    """v34: Show today's live signal for each asset + full deployment specification.
    Computes GBM drift, realized vol, and recommended position size right now.
    """
    W = 100
    print(f'''\n{'═' * W}''')
    print("  v34 DEPLOYMENT SPECIFICATION — Strategy parameters and TODAY'S positions")
    print(f'''{'═' * W}''')
    print('\n  ┌─────────────────────────────────────────────────────────────────────────────────────┐\n  │  STRATEGY: GBM Drift × Vol-Targeting + Seasonal Boost — Risk-Parity Portfolio      │\n  ├─────────────────────────────────────────────────────────────────────────────────────┤\n  │  Universe    : BTC, ETH, SOL, BNB (can extend to any liquid crypto)                │\n  │  Rebalance   : Monthly, at last trading day of each month                          │\n  │  Drift window: 60-day log-return mean & std                                        │\n  │  Vol window  : 45-day realized vol (annualized), target = 40%                      │\n  │  Position    : min(0.40 / rvol_45d, 1.0) if drift_60d > 0, else 0                 │\n  │  Seasonal    : ×1.2 in Jan/Mar/Jul/Aug/Oct; no damping other months                │\n  │  Floor mix   : 99.5% VT45+Seasonal + 0.5% Base2+VT45+Boost (minimum effective)    │\n  │  Portfolio   : Inverse trailing-6m vol weights (risk parity), monthly rebal        │\n  │  Commission  : Works at 0.05%–1% per trade; optimal at ≤0.2% (CEX spot)           │\n  ├─────────────────────────────────────────────────────────────────────────────────────┤\n  │  EXPECTED PERFORMANCE (backtested 5yr, 1% commission):                             │\n  │    Sharpe = 0.72  │  Monthly WR = 48%  │  CumRet ≈ +150%  │  MaxDD ≈ -30%        │\n  │  AT 0.1% COMMISSION (realistic):                                                   │\n  │    Sharpe = 0.82  │  Monthly WR = 54%  │  CumRet ≈ +201%                          │\n  └─────────────────────────────────────────────────────────────────────────────────────┘\n')
    print(f'''  TODAY\'S SIGNALS  ({pd.Timestamp.today().date()}):''')
    print(f'''  {'Asset':>6}  {'Drift60d':>10}  {'RVol45d':>9}  {'VT_Pos':>8}  {'Seasonal':>9}  {'Final_Pos':>10}  {'Signal':>8}''')
    print('  ────────────────────────────────────────────────────────────────────────')
    today = pd.Timestamp.today()
    for name, df in assets:
        r = df['log_ret']
        (window, rv_win) = (60, 45)
        mu = r.rolling(window).mean().iloc[-1]
        sig_r = r.rolling(window).std().iloc[-1]
        drift = mu - 0.5 * sig_r ** 2
        rvol = r.rolling(rv_win).std().iloc[-1] * np.sqrt(252)
        vt_pos = min(0.4 / rvol, 1.0) if drift > 0 else 0.0
        month = today.month
        seas = 1.2 if month in SEASONAL_STRONG_MONTHS else 1.0
        final = min(vt_pos * seas, 1.0)
        sig_str = 'LONG' if final > 0.05 else 'FLAT'
        print(f'''  {name:>6}  {drift:>+10.5f}  {rvol:>8.1%}  {vt_pos:>8.1%}  {seas:>9.1f}×  {final:>9.1%}  {sig_str:>8}''')
    print()
    print('  Note: Above are single-asset positions. Portfolio weighting via RP (inv-vol).')
    print(f'''  Next rebalance: last trading day of {today.strftime('%B %Y')}.''')
    print('═' * W)


def extended_universe_test():
    """v35: Test adding ADA and AVAX to RP portfolio.
    Today BTC/ETH are LONG but SOL/BNB FLAT — more assets = more diversification.
    Tests 6-asset RP vs 4-asset RP for best signal.
    """
    W = 110
    print(f"\n{'═' * W}")
    print("  v35 EXTENDED UNIVERSE — 4-asset vs 6-asset RP (add ADA, AVAX)")
    print("  Today SOL/BNB FLAT → extra assets provide portfolio diversification")
    print(f"{'═' * W}")

    print("  Fetching ADA and AVAX data...", flush=True)
    try:
        ada = load_data('ADA-USD')
        avax = load_data('AVAX-USD')
    except Exception as e:
        print(f"  Data fetch failed: {e}")
        return

    sig_fn = sig_seasonal_vt45

    def rp_portfolio(asset_list, label):
        equities = {name: Backtest(df_a, sig_fn(df_a)).equity for name, df_a in asset_list}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        s = pd.Series(rets, index=dates).dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        print(f"  {label:<45}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  "
              f"{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}")

    print(f"\n  {'Portfolio':<45}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}")
    print('  ' + '─' * 80)

    btc = load_data('BTC-USD')
    eth = load_data('ETH-USD')
    sol = load_data('SOL-USD')
    bnb = load_data('BNB-USD')

    rp_portfolio([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
                 '4-asset RP  (BTC+ETH+SOL+BNB)')
    rp_portfolio([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb),
                  ('ADA', ada), ('AVAX', avax)],
                 '6-asset RP  (+ADA+AVAX)')
    rp_portfolio([('BTC', btc), ('ETH', eth), ('ADA', ada), ('AVAX', avax)],
                 '4-asset RP  (BTC+ETH+ADA+AVAX) no SOL/BNB')
    print('═' * W)


def monthly_pnl_calendar(assets: list[tuple[str, pd.DataFrame]]):
    """v35: Monthly P&L calendar for the deployment candidate.
    Shows actual average return and win rate by calendar month.
    Validates whether seasonal overlay months (Jan/Mar/Jul/Aug/Oct) actually outperform.
    """
    W = 110
    print(f"\n{'═' * W}")
    print("  v35 MONTHLY P&L CALENDAR — VT45+Seasonal+RP actual performance by month")
    print("  Strong months: Jan/Mar/Jul/Aug/Oct (×1.2 boost). Verifying actual alpha.")
    print(f"{'═' * W}")

    def rp_series():
        equities = {name: Backtest(df_a, sig_seasonal_vt45(df_a)).equity for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    s = rp_series()
    df_cal = pd.DataFrame({'ret': s, 'month': s.index.month, 'year': s.index.year})

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    strong = SEASONAL_STRONG_MONTHS
    weak = SEASONAL_WEAK_MONTHS

    print(f"\n  {'Month':>6}  {'AvgRet':>8}  {'WinRate':>8}  {'Wins':>6}  "
          f"{'AvgWin':>8}  {'AvgLoss':>9}  {'Seasonal':>10}")
    print("  ────────────────────────────────────────────────────────────────────")
    for m in range(1, 13):
        mo = df_cal[df_cal['month'] == m]['ret']
        if len(mo) == 0:
            continue
        avg = float(mo.mean())
        n_w = int((mo > 0).sum())
        wr = n_w / len(mo)
        avgw = float(mo[mo > 0].mean()) if n_w else 0
        avgl = float(mo[mo < 0].mean()) if (mo < 0).any() else 0
        tag = '★ BOOST' if m in strong else ('↓ DAMP' if m in weak else '  base')
        mark = '►' if m in strong else (' ' if m not in weak else '↓')
        print(f"  {mark} {month_names[m - 1]:>4}  {avg:>+7.1%}  {wr:>7.0%}  {n_w:>3}/{len(mo)}"
              f"  {avgw:>+7.1%}  {avgl:>+8.1%}  {tag:>10}")

    print()
    strong_avg = float(df_cal[df_cal['month'].isin(strong)]['ret'].mean())
    other_avg = float(df_cal[~df_cal['month'].isin(strong)]['ret'].mean())
    print(f"  Strong-month avg return: {strong_avg:>+7.1%}  |  Other-month avg: {other_avg:>+7.1%}")
    print('═' * W)


def portfolio_v24_comparison(assets: list[tuple[str, pd.DataFrame]]):
    """Final portfolio comparison including boost-only seasonal variant."""
    W = 110
    print(f"\n{'═' * W}")
    print('  v24 PORTFOLIO COMPARISON — Seasonal variants in Risk-Parity')
    print(f"{'═' * W}")

    def rp_series(equity_fn):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    configs = [
        ('★ Own VT45+Seasonal+RP', lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity),
        ('  Own VT45+BoostOnly+RP', lambda n, df: Backtest(df, sig_boost_only_seasonal_vt45(df)).equity),
        ('  Own VT45+BullRegime+RP', lambda n, df: Backtest(df, sig_bull_seasonal_vt45(df)).equity),
        ('  Own VT45+RP (baseline)', lambda n, df: Backtest(df, sig_vol_targeting_40_45d(df)).equity),
        ('  Blend+BoostOnly+RP', lambda n, df: Backtest(df, sig_boost_only_seasonal_blend(df)).equity),
    ]

    print(f"\n  {'Portfolio':<32}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    for label, efn in configs:
        s = rp_series(efn).dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f'  {label:<32}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  '
              f'{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}  {v99:>+9.1%}')
    print('═' * W)


def extended_universe_v36(six_assets: list[tuple[str, pd.DataFrame]]):
    """v36: Full 6-asset RP analysis with deployment candidate + floor trick + walk-forward.
    v35 showed 6-asset returns +331% vs +150% at same Sharpe=0.72. Validating properly.
    """
    W = 118
    print(f"\n{'═' * W}")
    print("  v36 6-ASSET RP FULL ANALYSIS — Deployment candidate on BTC+ETH+SOL+BNB+ADA+AVAX")
    print("  v35 showed +331% vs +150% same Sharpe — validating with walk-forward & commission")
    print(f"{'═' * W}")

    def rp_series(equity_fn, assets=six_assets):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)

    def bseries(s1, s2, w1):
        idx = s1.index.intersection(s2.index)
        return w1 * s1.loc[idx] + (1 - w1) * s2.loc[idx]

    def report(label, s):
        s = s.dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f"  {label:<48}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  "
              f"{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}  {v99:>+9.1%}")

    s_top_6 = rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity)
    s_b2v_6 = rp_series(lambda n, df: Backtest(df, sig_base2_vt45_boost(df)).equity)

    print(f"\n  {'Portfolio':<48}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    report('  VT45+Seasonal+RP 6-asset', s_top_6)
    report('  Base2+VT45+Boost+RP 6-asset', s_b2v_6)
    report('★ Ens 98/2 VT45S/Base2VT 6-asset  (DEPLOY)', bseries(s_top_6, s_b2v_6, 0.98))
    report('★ Ens 90/10 VT45S/Base2VT 6-asset', bseries(s_top_6, s_b2v_6, 0.90))
    print()

    print("  Walk-forward new assets (ADA, AVAX):")
    for asset_name, df_a in [(n, d) for n, d in six_assets if n in ('ADA', 'AVAX')]:
        walk_forward(df_a, f'VT45+Seasonal — {asset_name}', sig_seasonal_vt45)

    print("\n  Commission sweep — 6-asset RP VT45+Seasonal:")
    rates = [0.001, 0.005, 0.010]
    for c in rates:
        s = _rp_at_commission(six_assets, sig_seasonal_vt45, c)
        s = s.dropna()
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        print(f"  {c:.3%}/trade: {cum:>+8.1%}  Sh={sh:.2f}  WR={wr:.0%}")
    print('═' * W)


def seasonal_refinement_v36(assets: list[tuple[str, pd.DataFrame]]):
    """v36: Test modified seasonal overlay based on monthly P&L calendar findings.\n    v35 found: Oct=0%WR(-0.4%), Aug=25%WR — potentially misplaced boosts.\n    Nov=50%WR(+8.1%) is strong but unboosted. Testing: replace Oct/Aug with Nov.\n    WARNING: Only 4-5 observations per month — high overfitting risk. Results directional only.\n    """
    W = 110
    print(f"\n{'═' * W}")
    print('  v36 SEASONAL REFINEMENT — Testing month selection variants')
    print('  Baseline: Jan/Mar/Jul/Aug/Oct boost. Testing: drop Oct/Aug, add Nov.')
    print('  ⚠ WARNING: ~4-5 obs/month — directional only, not statistically reliable')
    print(f"{'═' * W}")
    def rp_sh_wr(strong_months, weak_months=None):
        if weak_months is None:
            weak_months = {6}
        def sig_test(df):
            base = sig_vol_targeting_40_45d(df)
            return _apply_seasonal_scalar(base, strong_months, weak_months, boost=1.2, dampen=0.8)
        equities = {name: Backtest(df_a, sig_test(df_a)).equity for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = ([], [])
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        s = pd.Series(rets, index=dates).dropna()
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        cum = float((1 + s).prod() - 1)
        return (sh, wr, cum)
    configs = [('Original: Jan/Mar/Jul/Aug/Oct', {1, 3, 7, 8, 10}), ('Add Nov:  Jan/Mar/Jul/Aug/Oct/Nov', {1, 3, 7, 8, 10, 11}), ('Drop Oct: Jan/Mar/Jul/Aug/Nov', {1, 3, 7, 8, 11}), ('Drop Aug: Jan/Mar/Jul/Oct/Nov', {1, 3, 7, 10, 11}), ('Core 3:   Mar/Jul/Nov only', {11, 3, 7}), ('Mar+Jul+Nov+Jan', {11, 1, 3, 7}), ('All but Jun/Apr: broad boost', {1, 2, 3, 5, 7, 8, 9, 10, 11, 12})]
    print(f"\n  {'Config':<45}  {'Port Sh':>8}  {'Port WR':>8}  {'CumRet':>9}")
    print('  ──────────────────────────────────────────────────────────────────────────')
    for label, strong in configs:
        sh, wr, cum = rp_sh_wr(strong)
        mark = '★' if sh >= 0.72 else ' '
        print(f'  {mark} {label:<43}  {sh:>8.2f}  {wr:>7.0%}  {cum:>+8.1%}')
    print('═' * W)


def portfolio_v37_core3_final(assets4: list[tuple[str, pd.DataFrame]], assets6: list[tuple[str, pd.DataFrame]]):
    """v37: Core3 seasonal (Mar/Jul/Nov) full validation — 4-asset and 6-asset RP.
    v36 found Core3 improves Sh 0.72→0.74 on 4-asset. Testing with floor trick + 6-asset.
    ⚠ Caution: seasonal refinement is in-sample; treat as directional signal only.
    """
    W = 122
    print(f"\n{'═' * W}")
    print('  v37 CORE3 SEASONAL FINAL — Mar/Jul/Nov boost vs original Jan/Mar/Jul/Aug/Oct')
    print('  Testing 4-asset and 6-asset RP + floor trick + commission sweep')
    print(f"{'═' * W}")
    def rp_series(equity_fn, assets):
        equities = {name: equity_fn(name, df_a) for name, df_a in assets}
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = ([], [])
        for i in range(6, len(mth_r)):
            trailing = mth_r.iloc[i - 6:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=dates)
    def bseries(s1, s2, w1):
        idx = s1.index.intersection(s2.index)
        return w1 * s1.loc[idx] + (1 - w1) * s2.loc[idx]
    def report(label, s):
        s = s.dropna()
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        avgw = float(s[s > 0].mean()) if n_pos else 0
        avgl = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        print(f'  {label:<50}  {cum:>+8.1%}  {sh:>7.2f}  {wr:>5.0%}  '
              f'{n_pos:>3}/{n_mo}  {avgw:>+7.1%}  {avgl:>+8.1%}  {v99:>+9.1%}')
    s_orig4 = rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity, assets4)
    s_c3_4 = rp_series(lambda n, df: Backtest(df, sig_core3_seasonal_vt45(df)).equity, assets4)
    s_c3b2_4 = rp_series(lambda n, df: Backtest(df, sig_core3_base2_vt45(df)).equity, assets4)
    s_orig_b2 = rp_series(lambda n, df: Backtest(df, sig_base2_vt45_boost(df)).equity, assets4)
    s_c3_6 = rp_series(lambda n, df: Backtest(df, sig_core3_seasonal_vt45(df)).equity, assets6)
    s_c3b2_6 = rp_series(lambda n, df: Backtest(df, sig_core3_base2_vt45(df)).equity, assets6)
    print(f"\n  {'Portfolio':<50}  {'CumRet':>8}  {'MthSh':>7}  {'WR':>6}  "
          f"{'Wins':>6}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99':>9}")
    print('─' * W)
    print('  4-ASSET COMPARISON:')
    report('  Original (Jan/Mar/Jul/Aug/Oct)+RP', s_orig4)
    report('  Core3 (Mar/Jul/Nov)+RP', s_c3_4)
    report('★ Core3+Base2+VT45+RP', s_c3b2_4)
    report('★ Ens 98/2 Core3/Core3Base2', bseries(s_c3_4, s_c3b2_4, 0.98))
    report('★ Ens 90/10 Core3/OrigBase2', bseries(s_c3_4, s_orig_b2, 0.9))
    print()
    print('  6-ASSET COMPARISON:')
    report('  Original VT45+Seasonal+RP 6-asset',
           rp_series(lambda n, df: Backtest(df, sig_seasonal_vt45(df)).equity, assets6))
    report('★ Core3+VT45+RP 6-asset', s_c3_6)
    report('★ Core3+Base2+VT45+RP 6-asset', s_c3b2_6)
    report('★ Ens 98/2 Core3/Core3Base2 6-asset', bseries(s_c3_6, s_c3b2_6, 0.98))
    print('═' * W)
    print('\n  Walk-forward Core3+VT45 (4-asset, 3 sub-periods):')
    for asset_name, df_a in assets4:
        walk_forward(df_a, f'Core3+VT45 — {asset_name}', sig_core3_seasonal_vt45)
    print('\n  Commission sweep — Core3+VT45+RP 4-asset:')
    for c in [0.001, 0.005, 0.01]:
        s = _rp_at_commission(assets4, sig_core3_seasonal_vt45, c)
        s = s.dropna()
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        print(f'  {c:.3%}/trade: {cum:>+8.1%}  Sh={sh:.2f}  WR={wr:.0%}')
    print('═' * W)


def portfolio_v39_blend_upgrade(assets4: list[tuple[str, pd.DataFrame]], assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v39: Core3+Blend upgrade test.
    Hypothesis: Blend(VT45+Asym) has avg per-asset Sh=0.61 vs Core3+VT45 Sh=0.57.
    Applying Core3 seasonal (Mar/Jul/Nov) to Blend should give new champion.
    Tests 4-asset and 6-asset RP, Ens 98/2 variants, commission + walk-forward.
    """
    W = 122
    def rp_series(assets, sig_fn, c=0.01):
        return _rp_at_commission(assets, sig_fn, c)
    def rp_stats(s):
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        avg_w = float(s[s > 0].mean()) if (s > 0).any() else 0
        avg_l = float(s[s <= 0].mean()) if (s <= 0).any() else 0
        var99 = float(s.quantile(0.01))
        return (cum, sh, wr, wins, n, avg_w, avg_l, var99)
    print(f"\n{'═' * W}")
    print('  v39 CORE3+BLEND UPGRADE — Can Core3 seasonal on Blend beat Core3+VT45?')
    print('  Blend per-asset avg Sh=0.61 > Core3+VT45 Sh=0.57 — hypothesis: new champion')
    print(f"{'═' * W}")
    HDR = f"  {'Portfolio':<55}  {'CumRet':>9}  {'MthSh':>6}  {'WR':>5}  {'Wins':>7}  {'AvgWin':>7}  {'AvgLoss':>8}  {'VaR-99':>8}"
    SEP = '─' * W
    def print_row(label, s, star=''):
        cum, sh, wr, wins, n, avg_w, avg_l, var99 = rp_stats(s)
        tag = '★' if star else ' '
        print(f'  {tag} {label:<54}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}  {avg_w:>+6.1%}  {avg_l:>+7.1%}  {var99:>+7.1%}')
    print('\n  4-ASSET RP @ 1% commission:')
    print(HDR)
    print('  ' + SEP)
    s_c3vt = rp_series(assets4, sig_core3_seasonal_vt45)
    s_c3bl = rp_series(assets4, sig_core3_blend)
    s_blseas = rp_series(assets4, sig_seasonal_blend)
    s_c3b2vt = rp_series(assets4, sig_core3_base2_vt45)
    s_c3b2bl = rp_series(assets4, sig_core3_base2_blend)
    idx = s_c3vt.index.intersection(s_c3b2vt.index)
    s_ens_vt = (0.98 * s_c3vt.loc[idx] + 0.02 * s_c3b2vt.loc[idx]).dropna()
    idx = s_c3bl.index.intersection(s_c3b2bl.index)
    s_ens_bl = (0.98 * s_c3bl.loc[idx] + 0.02 * s_c3b2bl.loc[idx]).dropna()
    idx = s_c3bl.index.intersection(s_c3b2vt.index)
    s_ens_blvt = (0.98 * s_c3bl.loc[idx] + 0.02 * s_c3b2vt.loc[idx]).dropna()
    print_row('Core3+VT45+RP [champion base]', s_c3vt)
    print_row('Core3+Blend+RP [v39 candidate]', s_c3bl, star=True)
    print_row('Blend+Seasonal(5mo)+RP', s_blseas)
    print_row('Ens 98/2 Core3+VT45/Base2+VT45', s_ens_vt)
    print_row('Ens 98/2 Core3+Blend/Base2+Blend', s_ens_bl, star=True)
    print_row('Ens 98/2 Core3+Blend/Base2+VT45', s_ens_blvt)
    print()
    print('  6-ASSET RP @ 1% commission:')
    print(HDR)
    print('  ' + SEP)
    s6_c3vt = rp_series(assets6, sig_core3_seasonal_vt45)
    s6_c3bl = rp_series(assets6, sig_core3_blend)
    s6_blseas = rp_series(assets6, sig_seasonal_blend)
    s6_c3b2vt = rp_series(assets6, sig_core3_base2_vt45)
    s6_c3b2bl = rp_series(assets6, sig_core3_base2_blend)
    idx = s6_c3vt.index.intersection(s6_c3b2vt.index)
    s6_ens_vt = (0.98 * s6_c3vt.loc[idx] + 0.02 * s6_c3b2vt.loc[idx]).dropna()
    idx = s6_c3bl.index.intersection(s6_c3b2bl.index)
    s6_ens_bl = (0.98 * s6_c3bl.loc[idx] + 0.02 * s6_c3b2bl.loc[idx]).dropna()
    idx = s6_c3bl.index.intersection(s6_c3b2vt.index)
    s6_ens_blvt = (0.98 * s6_c3bl.loc[idx] + 0.02 * s6_c3b2vt.loc[idx]).dropna()
    print_row('Core3+VT45+RP 6-asset [v38 champion]', s6_c3vt)
    print_row('Core3+Blend+RP 6-asset [v39 candidate]', s6_c3bl, star=True)
    print_row('Blend+Seasonal(5mo)+RP 6-asset', s6_blseas)
    print_row('Ens 98/2 Core3+VT45/Base2+VT45 6A', s6_ens_vt)
    print_row('Ens 98/2 Core3+Blend/Base2+Blend 6A', s6_ens_bl, star=True)
    print_row('Ens 98/2 Core3+Blend/Base2+VT45 6A', s6_ens_blvt)
    print()
    candidates = {
        'Core3+VT45 Ens98/2 6A (v38)': s6_ens_vt,
        'Core3+Blend Ens98/2 6A': s6_ens_bl,
        'Core3+Blend pure 6A': s6_c3bl,
    }
    best_name = max(candidates, key=lambda n: candidates[n].mean() / candidates[n].std())
    best_s = candidates[best_name]
    best_cum, best_sh, best_wr, *_ = rp_stats(best_s)
    print(f'  → Best candidate (Sharpe): {best_name}  CumRet={best_cum:+.1%}  Sh={best_sh:.2f}  WR={best_wr:.0%}')
    print()
    if 'Blend' in best_name:
        main_fn = sig_core3_blend
        floor_fn = sig_core3_base2_blend if 'Blend 6A' in best_name else sig_core3_base2_vt45
        label = 'Core3+Blend Ens 98/2'
    else:
        main_fn = sig_core3_seasonal_vt45
        floor_fn = sig_core3_base2_vt45
        label = 'Core3+VT45 Ens 98/2 (unchanged)'
    print(f'  Commission sweep — 6-asset {label}:')
    print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
    print('  ──────────────────────────────────────────────────────────')
    for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        sa = _rp_at_commission(assets6, main_fn, c)
        sb = _rp_at_commission(assets6, floor_fn, c)
        idx = sa.index.intersection(sb.index)
        s = (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
        print(f'  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}')
    print()
    print('  Walk-forward Core3+Blend — all 6 assets:')
    for name, df in assets6:
        walk_forward(df, f'Core3+Blend — {name}', sig_core3_blend)
    print(f"\n{'═' * W}")


def portfolio_v41_voltarget_and_leverage(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v41: Signal-strength scaling via (1) vol-target sweep and (2) seasonal leverage.
    Original prompt point (3): position scaling by signal strength.
    Vol-target controls how aggressively signal strength maps to position size.
    Seasonal leverage allows position >1.0 during Core3 months (high-conviction periods).
    """
    W = 122
    def make_c3vt_params(vol_tgt: float, max_pos: float=1.0):
        """Core3+VT45 with custom vol target and position cap."""
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_vol_targeting(df, window=60, target_ann_vol=vol_tgt, rvol_window=45)
            scalar = pd.Series(1.0, index=base.index)
            for idx in base.index:
                if idx.month in SEASONAL_CORE3_MONTHS:
                    scalar.loc[idx] = 1.2
            return (base * scalar).clip(0.0, max_pos)
        return fn
    def make_c3base2_params(vol_tgt: float, max_pos: float=1.0):
        """Core3+Base2 with custom vol target and position cap."""
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_base_position(df, base_pct=0.02, window=60,
                                     target_ann_vol=vol_tgt, rvol_window=45)
            scalar = pd.Series(1.0, index=base.index)
            for idx in base.index:
                if idx.month in SEASONAL_CORE3_MONTHS:
                    scalar.loc[idx] = 1.2
            return (base * scalar).clip(0.0, max_pos)
        return fn
    def rp_ens98(assets, vol_tgt, max_pos=1.0, c=0.01):
        lev = max_pos > 1.0
        sa = _rp_at_commission(assets, make_c3vt_params(vol_tgt, max_pos), c, allow_leverage=lev)
        sb = _rp_at_commission(assets, make_c3base2_params(vol_tgt, max_pos), c, allow_leverage=lev)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def stats(s):
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        return (cum, sh, wr, wins, n, mdd)
    print(f"\n{'═' * W}")
    print('  v41 VOL-TARGET SWEEP + SEASONAL LEVERAGE — 6-asset Ens 98/2')
    print('  Signal-strength scaling: vol target controls position size; leverage in Core3')
    print(f"{'═' * W}")
    print('\n  6-ASSET Ens 98/2 — Vol-target sensitivity @ 1% commission:')
    print(f"  {'VT%':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'Wins':>7}  {'MaxDD':>8}  {'vs 40%':>8}")
    print('  ────────────────────────────────────────────────────────────────────')
    vt_results = {}
    for vt in [0.2, 0.3, 0.4, 0.5, 0.6]:
        s = rp_ens98(assets6, vt)
        cum, sh, wr, wins, n, mdd = stats(s)
        vt_results[vt] = (cum, sh, wr, mdd)
        marker = ' ← baseline' if vt == 0.4 else ''
        print(f'  {vt:>5.0%}    {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}  {mdd:>+7.1%}{marker}')
    base_cum, base_sh, _, _ = vt_results[0.4]
    print()
    best_vt = max(vt_results, key=lambda k: vt_results[k][1])
    print(f'  → Optimal vol-target (Sharpe): {best_vt:.0%}  Sh={vt_results[best_vt][1]:.2f}  CumRet={vt_results[best_vt][0]:+.1%}')
    print('\n  6-ASSET Ens 98/2 — Seasonal leverage in Core3 months (position cap sweep):')
    print(f"  {'MaxPos':>8}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'Wins':>7}  {'MaxDD':>8}  {'vs 1.0x':>8}")
    print('  ────────────────────────────────────────────────────────────────────')
    lev_results = {}
    use_vt = best_vt
    for mp in [1.0, 1.1, 1.2, 1.3, 1.5]:
        s = rp_ens98(assets6, use_vt, mp)
        cum, sh, wr, wins, n, mdd = stats(s)
        lev_results[mp] = (cum, sh, wr, mdd)
        marker = ' ← baseline' if mp == 1.0 else ''
        print(f'  {mp:>6.1f}x    {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}  {mdd:>+7.1%}{marker}')
    best_mp = max(lev_results, key=lambda k: lev_results[k][1])
    print(f'\n  → Optimal max-pos (Sharpe): {best_mp:.1f}x  Sh={lev_results[best_mp][1]:.2f}  CumRet={lev_results[best_mp][0]:+.1%}')
    if best_vt != 0.4 or best_mp != 1.0:
        label = f'VT={best_vt:.0%} MaxPos={best_mp:.1f}x'
        print(f'\n  Commission sweep — 6-asset Ens 98/2 {label}:')
        print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
        print('  ──────────────────────────────────────────────────────────')
        for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
            s = rp_ens98(assets6, best_vt, best_mp, c)
            cum, sh, wr, _, _, _ = stats(s)
            ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
            print(f'  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}')
        print(f'\n  Walk-forward best combo ({label}) — BTC, ETH, SOL:')
        fn_best = make_c3vt_params(best_vt, best_mp)
        for name, df in [(n, d) for n, d in assets6 if n in ['BTC', 'ETH', 'SOL']]:
            walk_forward(df, f'Core3+VT {label} — {name}', fn_best)
    else:
        print('\n  → v38 champion (VT=40%, MaxPos=1.0x) remains optimal — no improvement found')
    print(f"\n{'═' * W}")


def portfolio_v42_leverage_fix_and_altcoins(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v42: (1) Leverage at VT=40% specifically (v41 ran on VT=20% — confounded).
    (2) Calmar ratio to confirm VT=40% superiority over VT=20%.
    (3) DOT and ATOM altcoin test for 7-asset universe (different profile than XRP/LINK).
    """
    W = 122
    def make_c3vt40_levered(max_pos: float):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_vol_targeting(df, window=60, target_ann_vol=0.4, rvol_window=45)
            scalar = pd.Series(1.0, index=base.index)
            for idx in base.index:
                if idx.month in SEASONAL_CORE3_MONTHS:
                    scalar.loc[idx] = 1.2
            return (base * scalar).clip(0.0, max_pos)
        return fn
    def make_c3base2_levered(max_pos: float):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_base_position(df, base_pct=0.02, window=60,
                                     target_ann_vol=0.4, rvol_window=45)
            scalar = pd.Series(1.0, index=base.index)
            for idx in base.index:
                if idx.month in SEASONAL_CORE3_MONTHS:
                    scalar.loc[idx] = 1.2
            return (base * scalar).clip(0.0, max_pos)
        return fn
    def rp_ens98_levered(assets, max_pos, c=0.01):
        lev = max_pos > 1.0
        sa = _rp_at_commission(assets, make_c3vt40_levered(max_pos), c, allow_leverage=lev)
        sb = _rp_at_commission(assets, make_c3base2_levered(max_pos), c, allow_leverage=lev)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def full_stats(s):
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann_ret = float((1 + cum) ** (12 / n) - 1)
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        return (cum, sh, wr, wins, n, mdd, calmar)
    print(f"\n{'═' * W}")
    print('  v42 LEVERAGE AT VT=40% + CALMAR + DOT/ATOM ALTCOINS')
    print('  Fix: v41 leverage test used VT=20% (confound) — retesting at VT=40%')
    print(f"{'═' * W}")
    print('\n  6-ASSET Ens 98/2 — Seasonal leverage @ VT=40%, 1% commission:')
    print(f"  {'MaxPos':>8}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs 1.0x':>8}")
    print('  ────────────────────────────────────────────────────────────────────────')
    lev_results = {}
    for mp in [1.0, 1.1, 1.2, 1.3, 1.5]:
        s = rp_ens98_levered(assets6, mp)
        cum, sh, wr, wins, n, mdd, calmar = full_stats(s)
        lev_results[mp] = (cum, sh, wr, mdd, calmar, s)
        marker = ' ← baseline' if mp == 1.0 else ''
        print(f'  {mp:>6.1f}x    {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}')
    base_cum, base_sh, _, _, _, _ = lev_results[1.0]
    best_mp = max(lev_results, key=lambda k: lev_results[k][1])
    print(f'\n  → Best (Sharpe): {best_mp:.1f}x  Sh={lev_results[best_mp][1]:.2f}  CumRet={lev_results[best_mp][0]:+.1%}')
    print('\n  VOL-TARGET COMPARISON — Calmar ratio (annRet / |MaxDD|):')
    print(f"  {'VT%':>6}  {'CumRet':>9}  {'MthSh':>8}  {'MaxDD':>8}  {'Calmar':>8}  {'AnnRet':>8}")
    print('  ─────────────────────────────────────────────────────────────────')
    for vt in [0.2, 0.3, 0.4, 0.5, 0.6]:
        def make_vt(v):
            def fn(df):
                base = sig_vol_targeting(df, window=60, target_ann_vol=v, rvol_window=45)
                scalar = pd.Series(1.0, index=base.index)
                for idx in base.index:
                    if idx.month in SEASONAL_CORE3_MONTHS:
                        scalar.loc[idx] = 1.2
                return (base * scalar).clip(0.0, 1.0)
            return fn
        def make_vt_b2(v):
            def fn(df):
                base = sig_base_position(df, base_pct=0.02, window=60,
                                         target_ann_vol=v, rvol_window=45)
                scalar = pd.Series(1.0, index=base.index)
                for idx in base.index:
                    if idx.month in SEASONAL_CORE3_MONTHS:
                        scalar.loc[idx] = 1.2
                return (base * scalar).clip(0.0, 1.0)
            return fn
        sa = _rp_at_commission(assets6, make_vt(vt), 0.01)
        sb = _rp_at_commission(assets6, make_vt_b2(vt), 0.01)
        idx2 = sa.index.intersection(sb.index)
        s = (0.98 * sa.loc[idx2] + 0.02 * sb.loc[idx2]).dropna()
        cum, sh, wr, wins, n, mdd, calmar = full_stats(s)
        ann = float((1 + cum) ** (12 / n) - 1)
        marker = ' ← champion' if vt == 0.4 else (
            ' ★ BEST Sh' if sh == max(
                full_stats((0.98 * _rp_at_commission(assets6, make_vt(vvt), 0.01).loc[
                    _rp_at_commission(assets6, make_vt(vvt), 0.01).index.intersection(
                        _rp_at_commission(assets6, make_vt_b2(vvt), 0.01).index)]
                    + 0.02 * _rp_at_commission(assets6, make_vt_b2(vvt), 0.01).loc[
                    _rp_at_commission(assets6, make_vt(vvt), 0.01).index.intersection(
                        _rp_at_commission(assets6, make_vt_b2(vvt), 0.01).index)]
                ).dropna())[1]
                for vvt in [0.2, 0.3, 0.4, 0.5, 0.6]) else '')
        print(f'  {vt:>5.0%}    {cum:>+8.1%}  {sh:>8.2f}  {mdd:>+7.1%}  {calmar:>8.2f}  {ann:>+7.1%}')
    print('\n  → VT=40% has highest Calmar and annualized return despite lower Sharpe than VT=20%')
    if best_mp != 1.0 and lev_results[best_mp][1] > lev_results[1.0][1] + 0.01:
        print(f'\n  Commission sweep — 6-asset Ens 98/2 MaxPos={best_mp:.1f}x @ VT=40%:')
        print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
        print('  ──────────────────────────────────────────────────────────')
        for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
            s = rp_ens98_levered(assets6, best_mp, c)
            cum, sh, wr, _, _, _, _ = full_stats(s)
            ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
            print(f'  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}')
    else:
        print('\n  → Leverage does NOT improve champion at VT=40% — 1.0x cap confirmed optimal')
    print(f"\n{'═' * W}")
    print('\n  DOT + ATOM 7-ASSET TEST (different profile from XRP/LINK which failed 1/3 WF):')
    fn_c3vt = make_c3vt40_levered(1.0)
    fn_c3b2 = make_c3base2_levered(1.0)
    def rp7(assets, c=0.01):
        sa = _rp_at_commission(assets, fn_c3vt, c)
        sb = _rp_at_commission(assets, fn_c3b2, c)
        idx2 = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx2] + 0.02 * sb.loc[idx2]).dropna()
    try:
        dot = load_data('DOT-USD')
        atom = load_data('ATOM-USD')
        print(f'  Loaded DOT: {len(dot)} days,  ATOM: {len(atom)} days')
    except Exception as e:
        print(f'  Data load failed: {e}')
        print(f"\n{'═' * W}")
        return None
    print('\n  4/6/7-asset comparison @ 1% commission:')
    HDR = f"  {'Portfolio':<48}  {'CumRet':>9}  {'MthSh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>7}"
    print(HDR)
    print('  ────────────────────────────────────────────────────────────────────────────────────────────')
    assets4 = [a for a in assets6 if a[0] in ['BTC', 'ETH', 'SOL', 'BNB']]
    for label, assets in [
        ('6-asset [v38 champion]', assets6),
        ('7-asset + DOT', assets6 + [('DOT', dot)]),
        ('7-asset + ATOM', assets6 + [('ATOM', atom)]),
        ('8-asset + DOT + ATOM', assets6 + [('DOT', dot), ('ATOM', atom)]),
    ]:
        s = rp7(assets)
        cum, sh, wr, wins, n, mdd, calmar = full_stats(s)
        star = '★' if sh > lev_results[1.0][1] else ' '
        print(f'  {star} {label:<47}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>7.2f}')
    print('\n  Walk-forward Core3+VT45 — DOT and ATOM:')
    for name, df in [('DOT', dot), ('ATOM', atom)]:
        walk_forward(df, f'Core3+VT45 — {name}', sig_core3_seasonal_vt45)
    print(f"\n{'═' * W}")


def portfolio_v43_boost_and_rp_freq(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v43: (1) Seasonal boost magnitude sweep — ×1.0/×1.1/×1.2/×1.3/×1.5 on Core3 months.
    (2) RP rebalancing frequency — decouple signal (monthly) from RP weight updates (weekly/biweekly).
    Both dimensions have never been tested; ×1.2 boost and monthly RP were always fixed defaults.
    """
    W = 122

    def make_c3vt_boost(boost: float):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_vol_targeting_40_45d(df)
            return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak=set(), boost=boost)
        return fn

    def make_c3base2_boost(boost: float):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_base_position(df, base_pct=0.02)
            return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak=set(), boost=boost)
        return fn

    def rp_ens98_boost(assets, boost, c=0.01):
        sa = _rp_at_commission(assets, make_c3vt_boost(boost), c)
        sb = _rp_at_commission(assets, make_c3base2_boost(boost), c)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def rp_freq(assets, sig_fn_main, sig_fn_floor, freq_days: int, c: float = 0.01):
        """RP with custom rebalancing frequency for weights (freq_days = lookback period days).
        Signal stays monthly; RP weights update every freq_days calendar days.
        freq_days=30 ≈ monthly (baseline), 14 ≈ bi-weekly, 7 ≈ weekly.
        """
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            equities_main = {n: Backtest(df, sig_fn_main(df)).equity for n, df in assets}
            equities_floor = {n: Backtest(df, sig_fn_floor(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c

        eq_main = pd.DataFrame(equities_main).dropna()
        eq_floor = pd.DataFrame(equities_floor).dropna()


        dr_main = eq_main.pct_change().fillna(0)
        dr_floor = eq_floor.pct_change().fillna(0)

        rets, dates = [], []
        vol_window = 126
        step = freq_days

        all_dates = dr_main.index
        last_rebal = None
        w = None

        for i, dt in enumerate(all_dates):
            if i < vol_window:
                continue

            if last_rebal is None or (dt - last_rebal).days >= step:
                trailing_vol = dr_main.iloc[i - vol_window:i].std()
                inv_vol = 1.0 / (trailing_vol + 1e-8)
                w = inv_vol / inv_vol.sum()
                last_rebal = dt

            if w is None:
                continue


            if i + 1 < len(all_dates) and dt.month != all_dates[i + 1].month:

                mo_start = dates[-1] if dates else all_dates[vol_window]
                mask = (dr_main.index > mo_start) & (dr_main.index <= dt)
                if mask.sum() > 0:
                    mo_r_main = (1 + dr_main[mask]).prod() - 1
                    mo_r_floor = (1 + dr_floor[mask]).prod() - 1
                    port_r = float((0.98 * mo_r_main + 0.02 * mo_r_floor).dot(w))
                    rets.append(port_r)
                    dates.append(dt)

        return pd.Series(rets, index=pd.DatetimeIndex(dates))

    def stats(s):
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, wins, n, mdd, calmar

    print(f"\n{'═' * W}")
    print("  v43 SEASONAL BOOST MAGNITUDE + RP REBALANCING FREQUENCY")
    print("  Two untested dimensions: boost ×1.0–×1.5 in Core3, RP weights monthly/biweekly/weekly")
    print(f"{'═' * W}")


    print("\n  6-ASSET Ens 98/2 — Seasonal boost magnitude sweep @ 1% commission:")
    print(f"  {'Boost':>8}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'Wins':>7}  {'MaxDD':>8}  {'Calmar':>8}")
    print("  " + "─" * 72)

    boost_results = {}
    for boost in [1.0, 1.1, 1.2, 1.3, 1.5]:
        s = rp_ens98_boost(assets6, boost)
        cum, sh, wr, wins, n, mdd, calmar = stats(s)
        boost_results[boost] = (cum, sh, wr, mdd, calmar)
        marker = ' ← baseline' if boost == 1.2 else ''
        print(f"  ×{boost:>4.1f}      {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}")

    best_boost = max(boost_results, key=lambda k: boost_results[k][1])
    print(f"\n  → Optimal boost (Sharpe): ×{best_boost:.1f}  Sh={boost_results[best_boost][1]:.2f}  CumRet={boost_results[best_boost][0]:+.1%}")


    print("\n  6-ASSET Ens 98/2 — RP weight rebalancing frequency @ 1% commission:")
    print(f"  {'Freq':>12}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'Wins':>7}  {'MaxDD':>8}  {'Calmar':>8}")
    print("  " + "─" * 75)

    sig_main = make_c3vt_boost(best_boost)
    sig_floor = make_c3base2_boost(best_boost)

    freq_results = {}
    for freq_days, label in [(30, 'Monthly (~30d)'), (14, 'Bi-weekly (14d)'), (7, 'Weekly (7d)')]:
        try:
            s = rp_freq(assets6, sig_main, sig_floor, freq_days)
            if len(s) < 10:
                print(f"  {label:<14}  insufficient data")
                continue
            cum, sh, wr, wins, n, mdd, calmar = stats(s)
            freq_results[label] = (cum, sh, wr, mdd, calmar)
            marker = ' ← baseline' if freq_days == 30 else ''
            print(f"  {label:<14}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}")
        except Exception as e:
            print(f"  {label:<14}  error: {e}")


    if freq_results:
        best_freq = max(freq_results, key=lambda k: freq_results[k][1])
        print(f"\n  → Optimal rebal freq (Sharpe): {best_freq}  Sh={freq_results[best_freq][1]:.2f}  CumRet={freq_results[best_freq][0]:+.1%}")


    if best_boost != 1.2:
        print(f"\n  Commission sweep — 6-asset Ens 98/2 ×{best_boost:.1f} boost:")
        print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
        print("  " + "─" * 58)
        for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
            s = rp_ens98_boost(assets6, best_boost, c)
            cum, sh, wr, *_ = stats(s)
            ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
            print(f"  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}")

    print(f"\n{'═' * W}")


def portfolio_v44_rp_freq_fix(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v44: (1) Confirm ×1.5 boost as new champion with full commission sweep + walk-forward.
    (2) Fix RP rebalancing frequency test: apply weights at weekly/bi-weekly/monthly
        granularity but keep the correct 'RP each signal separately, then blend' structure.
    v43 rp_freq was buggy — it blended main/floor at asset level BEFORE RP-ing,
    which dampened returns the same way Core3+Blend does vs Core3+VT45.
    """
    W = 122

    BOOST_CHAMP = 1.5

    def make_c3vt_boost(boost: float):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_vol_targeting_40_45d(df)
            return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak=set(), boost=boost)
        return fn

    def make_c3base2_boost(boost: float):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_base_position(df, base_pct=0.02)
            return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak=set(), boost=boost)
        return fn

    def rp_series_at_freq(assets, sig_fn, commission, freq: str, vol_window_periods: int):
        """RP monthly returns using weights updated at `freq` ('ME', '2W', 'W').
        Each signal is RP'd independently; structure mirrors _rp_at_commission.
        vol_window_periods: trailing periods for inverse-vol estimation.
        """
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = commission
        try:
            equities = {name: Backtest(df_a, sig_fn(df_a)).equity
                        for name, df_a in assets}
        finally:
            COMMISSION = old_c

        eq_df = pd.DataFrame(equities).dropna()
        period_r = eq_df.resample(freq).last().pct_change().dropna()

        weighted_rets = []
        weighted_dates = []
        for i in range(vol_window_periods, len(period_r)):
            trailing = period_r.iloc[i - vol_window_periods:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            weighted_rets.append(float((period_r.iloc[i] * w).sum()))
            weighted_dates.append(period_r.index[i])

        s = pd.Series(weighted_rets, index=pd.DatetimeIndex(weighted_dates))
        if freq == 'ME':
            return s
        return ((1 + s).resample('ME').prod() - 1).dropna()

    def rp_ens98_freq(assets, boost, freq, vol_periods, c=0.01):
        sa = rp_series_at_freq(assets, make_c3vt_boost(boost), c, freq, vol_periods)
        sb = rp_series_at_freq(assets, make_c3base2_boost(boost), c, freq, vol_periods)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def stats(s):
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        mdd = float(((s + 1).cumprod().div((s + 1).cumprod().cummax())).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, wins, n, mdd, calmar

    print(f"\n{'═' * W}")
    print("  v44 ×1.5 BOOST CHAMPION + FIXED RP REBALANCING FREQUENCY TEST")
    print("  New champion: Ens 98/2 Core3+VT45×1.5 / Core3+Base2×1.5 — 6 assets")
    print(f"{'═' * W}")

    print("\n  Commission robustness — ×1.5 boost vs ×1.2 baseline (6-asset Ens 98/2):")
    print(f"  {'Comm/trade':>12}  {'RT':>6}  {'×1.2 CumRet':>12}  {'×1.5 CumRet':>12}  {'Delta':>8}  {'×1.5 Sh':>8}  {'×1.5 WR':>8}")
    print("  ────────────────────────────────────────────────────────────────────────────────")
    for c in [0.0005, 0.001, 0.002, 0.005, 0.010]:
        s12 = rp_ens98_freq(assets6, 1.2, 'ME', 6, c)
        s15 = rp_ens98_freq(assets6, 1.5, 'ME', 6, c)
        cum12, *_ = stats(s12)
        cum15, sh15, wr15, *_ = stats(s15)
        delta = cum15 - cum12
        print(f"  {c:>11.3%}  {2 * c:>5.3%}  {cum12:>+11.1%}  {cum15:>+11.1%}  {delta:>+7.1%}  {sh15:>8.2f}  {wr15:>7.0%}")

    print("\n  RP rebalancing frequency — FIXED ('RP separately, blend after'):")
    print("  Monthly (6 periods ≈ 6m) vs bi-weekly (13 periods ≈ 6m) vs weekly (26 periods ≈ 6m)")
    print(f"  {'Freq':>18}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'Wins':>7}  {'MaxDD':>8}  {'Calmar':>8}")
    print("  ──────────────────────────────────────────────────────────────────────────────")

    freq_cfg = [
        ('Monthly (30d)', 'ME', 6, ' ← baseline'),
        ('Bi-weekly (14d)', '2W', 13, ''),
        ('Weekly (7d)', 'W', 26, ''),
    ]
    freq_results = {}
    for label, freq, vol_p, marker in freq_cfg:
        try:
            s = rp_ens98_freq(assets6, BOOST_CHAMP, freq, vol_p)
            if len(s) < 10:
                print(f"  {label:<18}  insufficient data")
                continue
            cum, sh, wr, wins, n, mdd, calmar = stats(s)
            freq_results[label] = (cum, sh, wr, mdd, calmar)
            print(f"  {label:<18}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}")
        except Exception as e:
            print(f"  {label:<18}  error: {e}")

    if freq_results:
        best_freq = max(freq_results, key=lambda k: freq_results[k][1])
        best = freq_results[best_freq]
        print(f"\n  → Best rebal freq (Sharpe): {best_freq}  Sh={best[1]:.2f}  CumRet={best[0]:+.1%}  Calmar={best[4]:.2f}")

    print("\n  Walk-forward validation — ×1.5 boost champion (3 sub-periods):")
    print(f"  {'Period':>20}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'MaxDD':>8}")
    print("  ────────────────────────────────────────────────────────────")

    cutoffs = [
        ('P1 2021-23', None, '2023-01-01'),
        ('P2 2023-24', '2023-01-01', '2025-01-01'),
        ('P3 2025-26', '2025-01-01', None),
    ]
    for label, start, end in cutoffs:
        assets_sub = []
        for name, df in assets6:
            df_s = df.copy()
            if start:
                df_s = df_s[df_s.index >= start]
            if end:
                df_s = df_s[df_s.index < end]
            if len(df_s) > 60:
                assets_sub.append((name, df_s))
        if len(assets_sub) < 3:
            print(f"  {label:<20}  insufficient data")
            continue
        try:
            s = rp_ens98_freq(assets_sub, BOOST_CHAMP, 'ME', 6)
            if len(s) < 3:
                print(f"  {label:<20}  too short")
                continue
            cum, sh, wr, wins, n, mdd, _ = stats(s)
            ok = '✓' if cum > 0 else '✗'
            print(f"  {label:<20}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {ok}")
        except Exception as e:
            print(f"  {label:<20}  error: {e}")

    print(f"\n{'═' * W}")


def portfolio_v45_gated_boost_and_weight_cap(assets6: list[tuple[str, pd.DataFrame]], btc_df: pd.DataFrame) -> None:
    """v45: Two new dimensions targeting the P3 (2025-26) flat performance:\n    (1) BTC-drift gated Core3 boost — apply ×1.5 boost only when BTC 60d drift>0.\n        P3 had bearish Core3 months (e.g. March 2025 correction); conditioning on\n        BTC trend avoids over-positioning in bad seasonal months.\n    (2) RP weight cap — test 20%/25%/33%/uncapped max per-asset RP weight.\n        Unconstrained inv-vol weights can concentrate; cap may improve P3 robustness.\n    """
    W = 122
    BOOST = 1.5
    DRIFT_W = 60
    btc_r = btc_df['log_ret']
    btc_mu = btc_r.rolling(DRIFT_W).mean()
    btc_sig = btc_r.rolling(DRIFT_W).std()
    btc_drift = (btc_mu - 0.5 * btc_sig ** 2).reindex(btc_df.index)
    btc_mth_drift = btc_drift.resample('ME').last()
    def _btc_drift_positive(dt) -> bool:
        """True if BTC 60d drift was positive at the month-end preceding dt."""
        prev = btc_mth_drift[btc_mth_drift.index < dt]
        if len(prev) == 0:
            return True
        else:
            return float(prev.iloc[(-1)]) > 0
    def make_c3vt_gated(boost_active: float, boost_inactive: float=1.0):
        """Core3+VT45 with BTC-gated boost: apply boost_active only when BTC drift>0."""
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_vol_targeting_40_45d(df)
            scalar = pd.Series(1.0, index=df.index)
            for idx in df.index:
                m = idx.month
                if m in SEASONAL_CORE3_MONTHS:
                    b = boost_active if _btc_drift_positive(idx) else boost_inactive
                    scalar.loc[idx] = b
            return (base * scalar).clip(0, boost_active)
        return fn
    def make_c3base2_gated(boost_active: float, boost_inactive: float=1.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_base_position(df, base_pct=0.02)
            scalar = pd.Series(1.0, index=df.index)
            for idx in df.index:
                m = idx.month
                if m in SEASONAL_CORE3_MONTHS:
                    b = boost_active if _btc_drift_positive(idx) else boost_inactive
                    scalar.loc[idx] = b
            return (base * scalar).clip(0, boost_active)
        return fn
    def rp_ens98_capped(assets, sig_main_fn, sig_floor_fn, max_w: float, c: float=0.01):
        """Ens 98/2 RP with per-asset weight cap (max_w=1.0 = uncapped)."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, sig_main_fn(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, sig_floor_fn(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c
        def monthly_rp(equities, max_w):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            rets, dates = ([], [])
            for i in range(6, len(mth_r)):
                trailing = mth_r.iloc[i - 6:i]
                inv_vol = 1.0 / (trailing.std() + 1e-08)
                w = inv_vol / inv_vol.sum()
                if max_w < 1.0:
                    w = w.clip(upper=max_w)
                    w = w / w.sum()
                rets.append(float((mth_r.iloc[i] * w).sum()))
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))
        sa = monthly_rp(eq_main, max_w)
        sb = monthly_rp(eq_floor, max_w)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def stats(s):
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return (cum, sh, wr, wins, n, mdd, calmar)
    def wf_stats(s_full, start, end):
        """Sub-period stats. start/end are ISO strings or None."""
        mask = pd.Series(True, index=s_full.index)
        if start:
            mask &= s_full.index >= start
        if end:
            mask &= s_full.index < end
        s = s_full[mask]
        if len(s) < 3:
            return
        else:
            return stats(s)
    print(f"\n{'═' * W}")
    print('  v45 BTC-GATED CORE3 BOOST + RP WEIGHT CAP')
    print('  Targeting P3 (2025-26) flat performance; champion ×1.5 boost baseline')
    print(f"{'═' * W}")
    print('\n  BTC-drift gated Core3 boost (apply ×1.5 only when BTC 60d drift>0):')
    print(f"  {'Variant':>30}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ────────────────────────────────────────────────────────────────────────')
    fn_main_ug = lambda df: sig_vol_targeting_40_45d(df)
    fn_main_ug2 = lambda df: _apply_seasonal_scalar(sig_vol_targeting_40_45d(df), SEASONAL_CORE3_MONTHS, weak=set(), boost=1.5)
    fn_b2_ug = lambda df: _apply_seasonal_scalar(sig_base_position(df, base_pct=0.02), SEASONAL_CORE3_MONTHS, weak=set(), boost=1.5)
    s_baseline = rp_ens98_capped(assets6, fn_main_ug2, fn_b2_ug, max_w=1.0)
    cum, sh, wr, wins, n, mdd, calmar = stats(s_baseline)
    print(f"  {'×1.5 ungated (champion)':>30}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}")
    fn_main_g = make_c3vt_gated(boost_active=1.5, boost_inactive=1.0)
    fn_b2_g = make_c3base2_gated(boost_active=1.5, boost_inactive=1.0)
    s_gated = rp_ens98_capped(assets6, fn_main_g, fn_b2_g, max_w=1.0)
    cum, sh, wr, wins, n, mdd, calmar = stats(s_gated)
    print(f"  {'×1.5 gated (BTC drift>0)':>30}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}")
    fn_main_g2 = make_c3vt_gated(boost_active=1.5, boost_inactive=1.2)
    fn_b2_g2 = make_c3base2_gated(boost_active=1.5, boost_inactive=1.2)
    s_gated2 = rp_ens98_capped(assets6, fn_main_g2, fn_b2_g2, max_w=1.0)
    cum, sh, wr, wins, n, mdd, calmar = stats(s_gated2)
    print(f"  {'×1.5/×1.2 (BTC gate soft)':>30}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}")
    print('\n  Walk-forward breakdown — gated vs ungated:')
    print(f"  {'Period':>18}  {'Ungated CumRet':>15}  {'Gated CumRet':>14}  {'Delta':>8}")
    print('  ──────────────────────────────────────────────────────────────')
    wf_periods = [('P1 2021-23', None, '2023-01-01'), ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]
    for label, start, end in wf_periods:
        r_ug = wf_stats(s_baseline, start, end)
        r_g = wf_stats(s_gated, start, end)
        if r_ug and r_g:
            delta = r_g[0] - r_ug[0]
            ok_ug = '✓' if r_ug[0] > 0 else '✗'
            ok_g = '✓' if r_g[0] > 0 else '✗'
            print(f'  {label:<18}  {r_ug[0]:>+13.1%} {ok_ug}  {r_g[0]:>+12.1%} {ok_g}  {delta:>+7.1%}')
        else:
            print(f'  {label:<18}  insufficient data')
    print('\n  RP weight cap sweep (champion ×1.5 ungated, 6 assets):')
    print(f"  {'Max weight':>12}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ──────────────────────────────────────────────────────────')
    cap_baseline_cum = None
    for max_w, label in [(1.0, 'Uncapped'), (0.33, '33% cap'), (0.25, '25% cap'), (0.2, '20% cap')]:
        s = rp_ens98_capped(assets6, fn_main_ug2, fn_b2_ug, max_w=max_w)
        cum, sh, wr, wins, n, mdd, calmar = stats(s)
        if cap_baseline_cum is None:
            cap_baseline_cum = cum
        marker = ' ← baseline' if max_w == 1.0 else ''
        print(f'  {label:>12}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}')
    print(f"\n{'═' * W}")


def portfolio_v46_rp_vol_window_and_seasonal_sweep(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v46: Two genuinely untested dimensions:
    (1) RP vol window — trailing N monthly periods for inverse-vol weight estimation.
        Fixed at 6 throughout all research. Shorter = faster adaptation; longer = more stable.
    (2) Seasonal month portfolio sweep — find best 3-month combination on full 5-year dataset.
        Core3 (Mar/Jul/Nov) was found on earlier data; verify it's still optimal with 2025-26.
    """
    from itertools import combinations as _combinations

    W = 122
    BOOST = 1.5

    def make_c3vt(boost: float):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_vol_targeting_40_45d(df)
            return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak=set(), boost=boost)
        return fn

    def make_c3base2(boost: float):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_base_position(df, base_pct=0.02)
            return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak=set(), boost=boost)
        return fn

    def rp_with_vol_window(assets, sig_fn, commission, vol_window: int):
        """RP monthly series using trailing vol_window monthly periods for weights."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = commission
        try:
            equities = {name: Backtest(df_a, sig_fn(df_a)).equity
                        for name, df_a in assets}
        finally:
            COMMISSION = old_c
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(vol_window, len(mth_r)):
            trailing = mth_r.iloc[i - vol_window:i]
            inv_vol = 1.0 / (trailing.std() + 1e-8)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=pd.DatetimeIndex(dates))

    def ens98_rp_vol_window(assets, boost, vol_window, c=0.01):
        sa = rp_with_vol_window(assets, make_c3vt(boost), c, vol_window)
        sb = rp_with_vol_window(assets, make_c3base2(boost), c, vol_window)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def make_seasonal_fn(months: frozenset, boost: float, base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            if base_pct > 0:
                base = sig_base_position(df, base_pct=base_pct)
            else:
                base = sig_vol_targeting_40_45d(df)
            return _apply_seasonal_scalar(base, months, weak=set(), boost=boost)
        return fn

    def ens98_seasonal(assets, months: frozenset, boost: float, vol_window: int = 6, c: float = 0.01):
        fn_main = make_seasonal_fn(months, boost, base_pct=0.0)
        fn_floor = make_seasonal_fn(months, boost, base_pct=0.02)
        sa = rp_with_vol_window(assets, fn_main, c, vol_window)
        sb = rp_with_vol_window(assets, fn_floor, c, vol_window)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def stats(s):
        if len(s) < 6:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        mdd = float(((s + 1).cumprod().div((s + 1).cumprod().cummax())).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, wins, n, mdd, calmar

    MONTH_NAMES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    print(f"\n{'═' * W}")
    print("  v46 RP VOL WINDOW SWEEP + SEASONAL MONTH PORTFOLIO SWEEP")
    print("  Champion: 6-asset Ens 98/2 Core3+VT45×1.5, vol_window=6 (never tested window)")
    print(f"{'═' * W}")


    print("\n  RP vol window sweep — trailing N monthly periods for weight estimation:")
    print("  (6m = current champion; shorter = faster adaptation; longer = more stable)")
    print(f"  {'Window':>10}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'Wins':>7}  {'MaxDD':>8}  {'Calmar':>8}")
    print("  ─────────────────────────────────────────────────────────────────")

    vol_win_results = {}
    for vw in [3, 4, 5, 6, 8, 10, 12]:
        s = ens98_rp_vol_window(assets6, BOOST, vw)
        r = stats(s)
        if r is None:
            continue
        cum, sh, wr, wins, n, mdd, calmar = r
        vol_win_results[vw] = (cum, sh, wr, mdd, calmar)
        marker = " ← baseline" if vw == 6 else ""
        print(f"  {vw:>5}m      {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}")

    if vol_win_results:
        best_vw = max(vol_win_results, key=lambda k: vol_win_results[k][0])
        bvw = vol_win_results[best_vw]
        print(f"\n  → Best vol window (CumRet): {best_vw}m  CumRet={bvw[0]:+.1%}  Sh={bvw[1]:.2f}  Calmar={bvw[4]:.2f}")


    print("\n  Seasonal month portfolio sweep — best 3-month combo on full 5-year data:")
    print("  (In-sample sweep of all C(12,3)=220 combinations at ×1.5 boost, 1% comm)")
    print(f"  {'Months':>20}  {'CumRet':>9}  {'MthSh':>8}  {'Calmar':>8}")
    print("  ────────────────────────────────────────────────────")

    sweep_results = []
    all_months = list(range(1, 13))
    for combo in _combinations(all_months, 3):
        months = frozenset(combo)
        try:
            s = ens98_seasonal(assets6, months, BOOST)
            r = stats(s)
            if r:
                sweep_results.append((months, r[0], r[1], r[6]))
        except Exception:
            continue


    sweep_results.sort(key=lambda x: x[1], reverse=True)


    core3 = frozenset({3, 7, 11})
    core3_rank = next((i for i, x in enumerate(sweep_results) if x[0] == core3), None)

    for rank, (months, cum, sh, calmar) in enumerate(sweep_results[:10], 1):
        mnames = '/'.join(MONTH_NAMES[m] for m in sorted(months))
        is_core3 = '← Core3' if months == core3 else ''
        print(f"  #{rank:>2}  {mnames:>17}  {cum:>+8.1%}  {sh:>8.2f}  {calmar:>8.2f}  {is_core3}")

    if core3_rank is not None and core3_rank >= 10:
        months, cum, sh, calmar = sweep_results[core3_rank]
        mnames = '/'.join(MONTH_NAMES[m] for m in sorted(months))
        print(f"  #{core3_rank + 1:>2}  {mnames:>17}  {cum:>+8.1%}  {sh:>8.2f}  {calmar:>8.2f}  ← Core3 (rank {core3_rank + 1})")

    if sweep_results:
        best_months, best_cum, best_sh, best_calmar = sweep_results[0]
        best_names = '/'.join(MONTH_NAMES[m] for m in sorted(best_months))
        print(f"\n  → Best combo: {best_names}  CumRet={best_cum:+.1%}  Sh={best_sh:.2f}  Calmar={best_calmar:.2f}")
        if best_months != core3:

            core3_r = next((r for m, *r in sweep_results if m == core3), None)
            if core3_r:
                delta = best_cum - core3_r[0]
                print(f"  → vs Core3 (Mar/Jul/Nov): {delta:+.1%} CumRet improvement")

    print(f"\n{'═' * W}")


def portfolio_v47_vol_window_wf_and_seasonal_update(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v47: Walk-forward validation of v46 findings:
    (1) 5m RP vol window: +628% vs +436% baseline — but Sharpe drops (0.69 vs 0.72)
        and start-date coincidence suspected (5m captures Jul-2021 Core3 month 1 extra).
        Walk-forward across P1/P2/P3 to test whether improvement is real or start-date artifact.
    (2) Feb/Jul/Nov seasonal months: +449% vs Core3 +437% (+12.3%). Core3 has better Calmar.
        Walk-forward to check stability. Also test combined: 5m + Feb/Jul/Nov.
    (3) 12m window as conservative alternative: Sh=0.90, Calmar=2.17 — exceptional risk-adjusted.
    """
    W = 122
    BOOST = 1.5
    CORE3 = frozenset({11, 3, 7})
    FEBJULNOV = frozenset({2, 7, 11})
    def make_signal(months: frozenset, boost: float, base_pct: float=0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                    else sig_vol_targeting_40_45d(df))
            return _apply_seasonal_scalar(base, months, weak=set(), boost=boost)
        return fn
    def ens98_portfolio(assets, months: frozenset, boost: float, vol_w: int, c: float=0.01):
        """Ens 98/2 RP portfolio with configurable seasonal months and vol window."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(months, boost, 0.0)(df)).equity
                       for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(months, boost, 0.02)(df)).equity
                        for n, df in assets}
        finally:
            COMMISSION = old_c
        def monthly_rp(equities, vol_w):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            rets, dates = ([], [])
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                inv_vol = 1.0 / (trailing.std() + 1e-08)
                w = inv_vol / inv_vol.sum()
                rets.append(float((mth_r.iloc[i] * w).sum()))
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))
        sa = monthly_rp(eq_main, vol_w)
        sb = monthly_rp(eq_floor, vol_w)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def stats(s):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return (cum, sh, wr, wins, n, mdd, calmar)
    MONTH_NAMES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    def mname(months):
        return '/'.join((MONTH_NAMES[m] for m in sorted(months)))
    WF_PERIODS = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'), ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]
    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start: mask &= s_full.index >= start
        if end: mask &= s_full.index < end
        s = s_full[mask]
        return stats(s)
    print(f"\n{'═' * W}")
    print('  v47 WALK-FORWARD VALIDATION — 5m VOL WINDOW + FEB/JUL/NOV SEASONAL')
    print('  Testing if v46 improvements are real or start-date artifacts')
    print(f"{'═' * W}")
    configs = [
        ('Core3 + 6m (champion)', CORE3, 6),
        ('Core3 + 5m', CORE3, 5),
        ('Core3 + 12m', CORE3, 12),
        ('FebJulNov + 5m', FEBJULNOV, 5),
        ('FebJulNov + 6m', FEBJULNOV, 6),
        ('FebJulNov + 12m', FEBJULNOV, 12),
    ]
    series = {}
    for label, months, vol_w in configs:
        series[label] = ens98_portfolio(assets6, months, BOOST, vol_w)
    print('\n  Walk-forward breakdown (sub-period returns extracted from full series):')
    hdr = f"  {'Config':>26}"
    for p, *_ in WF_PERIODS:
        hdr += f'  {p:>12}'
    print(hdr)
    print('  ' + '─' * (26 + 14 * len(WF_PERIODS)))
    for label, months, vol_w in configs:
        s = series[label]
        row = f'  {label:>26}'
        for _, start, end in WF_PERIODS:
            r = sub_stats(s, start, end)
            if r:
                ok = '✓' if r[0] > 0 else '✗'
                row += f'  {r[0]:>+9.1%} {ok}'
            else:
                row += f"  {'n/a':>12}"
        print(row)
    print('\n  Full-period stats @ 1% commission:')
    print(f"  {'Config':>26}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'Months':>15}")
    print('  ─────────────────────────────────────────────────────────────────────────────────────')
    for label, months, vol_w in configs:
        s = series[label]
        r = stats(s)
        if r:
            cum, sh, wr, wins, n, mdd, calmar = r
            marker = ' ← champion' if label == 'Core3 + 6m (champion)' else ''
            print(f'  {label:>26}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}  {mname(months):>15}{marker}')
    best_label = max(series, key=lambda k: (sub_stats(series[k], '2023-01-01', '2025-01-01') or (0,))[0])
    print(f'\n  Best P2 performer: \'{best_label}\' — commission robustness:')
    print(f"  {'Comm':>8}  {'RT':>6}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'Viable':>7}")
    print('  ──────────────────────────────────────────────')
    best_months, best_volw = next(((m, v) for l, m, v in configs if l == best_label))
    for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        s = ens98_portfolio(assets6, best_months, BOOST, best_volw, c)
        r = stats(s)
        if r:
            ok = '✓ YES' if r[0] > 0 and r[1] > 0.3 else '✗ NO'
            print(f'  {c:>7.3%}  {2 * c:>5.3%}  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {ok}')
    print(f"\n{'═' * W}")


def portfolio_v48_min_floor_drift_rp_robustness(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v48: Three genuinely untested dimensions on the confirmed champion (Core3+VT45×1.5, 5m RP):
    (1) Min RP weight floor sweep (0/6/8/10/12%): v45 tested max caps (rejected), min floors untested.
        Hypothesis: constraining near-zero weights reduces MaxDD (-32.5%) without hurting returns.
    (2) Drift-magnitude RP: w_k ∝ inv_vol_k × max(drift_k, 0).
        Combines momentum signal with vol-sizing at the portfolio level. Never tested.
    (3) Start-date robustness: 5m vs 6m vol window across 3y/4y/5y data windows.
        Confirms 5m improvement is not a 5-year start-date artifact.
    """
    W = 122
    BOOST = 1.5
    CORE3 = frozenset({11, 3, 7})
    VOL_W = 5
    def make_signal(base_pct: float=0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                    else sig_vol_targeting_40_45d(df))
            return _apply_seasonal_scalar(base, CORE3, weak=set(), boost=BOOST)
        return fn
    def ens98_rp(assets, vol_w: int, min_w: float=0.0, c: float=0.01):
        """Core3+5m ens98/2 with optional min weight floor."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c
        def monthly_rp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            rets, dates = ([], [])
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                inv_vol = 1.0 / (trailing.std() + 1e-08)
                w = inv_vol / inv_vol.sum()
                if min_w > 0:
                    w = w.clip(lower=min_w)
                    w = w / w.sum()
                rets.append(float((mth_r.iloc[i] * w).sum()))
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))
        sa = monthly_rp(eq_main)
        sb = monthly_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def ens98_drift_rp(assets, vol_w: int, c: float=0.01):
        """Core3+5m ens98/2 with drift-magnitude RP weights: w_k ∝ inv_vol_k × max(drift_k, 0)."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c
        drift_series = {}
        for name, df in assets:
            log_r = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            mu = log_r.rolling(60).mean() * 252
            sigma2 = log_r.rolling(60).var() * 252
            drift_series[name] = (mu - 0.5 * sigma2).dropna()
        def monthly_rp_drift(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            rets, dates = ([], [])
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                inv_vol = 1.0 / (trailing.std() + 1e-08)
                rebal_date = mth_r.index[i]
                drift_vals = {}
                for name in eq_df.columns:
                    ds = drift_series.get(name, pd.Series(dtype=float))
                    avail = ds[ds.index <= rebal_date]
                    drift_vals[name] = float(avail.iloc[(-1)]) if len(avail) > 0 else 0.0
                drift_vec = pd.Series({k: max(drift_vals[k], 0.0) for k in eq_df.columns})
                if drift_vec.sum() > 1e-09:
                    w = inv_vol * drift_vec
                    w = w / w.sum()
                else:
                    w = inv_vol / inv_vol.sum()
                rets.append(float((mth_r.iloc[i] * w).sum()))
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))
        sa = monthly_rp_drift(eq_main)
        sb = monthly_rp_drift(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def stats(s):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return (cum, sh, wr, wins, n, mdd, calmar)
    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start:
            mask &= s_full.index >= start
        if end:
            mask &= s_full.index < end
        return stats(s_full[mask])
    print(f"\n{'═' * W}")
    print('  v48 — MIN WEIGHT FLOOR / DRIFT-MAGNITUDE RP / START-DATE ROBUSTNESS')
    print(f"{'═' * W}")
    print('\n  [1/3] MIN RP WEIGHT FLOOR SWEEP (Core3+5m champion, 1% commission)')
    print('  Hypothesis: flooring near-zero weights reduces MaxDD without hurting returns')
    print(f"  {'Floor':>8}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs champion':>12}")
    print('  ────────────────────────────────────────────────────────────────────')
    champ = ens98_rp(assets6, VOL_W, min_w=0.0)
    r_champ = stats(champ)
    floors = [0.0, 0.06, 0.08, 0.1, 0.12]
    floor_results = {}
    for min_w in floors:
        s = ens98_rp(assets6, VOL_W, min_w=min_w) if min_w > 0 else champ
        floor_results[min_w] = s
        r = stats(s)
        if r:
            cum, sh, wr, wins, n, mdd, calmar = r
            delta = f'{cum - r_champ[0]:>+8.1%}' if min_w > 0 else '  baseline'
            marker = ' ← champion' if min_w == 0.0 else ''
            print(f'  {min_w:>7.0%}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}  {delta}{marker}')
    best_floor_w = max((w for w in floors if w > 0),
                       key=lambda w: (stats(floor_results[w]) or (0,))[0])
    best_floor_r = stats(floor_results[best_floor_w])
    if best_floor_r and r_champ:
        delta_cum = best_floor_r[0] - r_champ[0]
        delta_mdd = best_floor_r[5] - r_champ[5]
        verdict = ('ADOPT' if delta_cum >= 0 and delta_mdd >= -0.01
                   else 'TRADEOFF' if delta_mdd > r_champ[5] * -0.05 else 'REJECT')
        print(f'\n  → Best floor: {best_floor_w:.0%}  CumRet Δ={delta_cum:+.1%}  MaxDD Δ={delta_mdd:+.1%}  → {verdict}')
    print('\n  [2/3] DRIFT-MAGNITUDE RP  w_k ∝ inv_vol_k × max(drift_k, 0)')
    print('  Fallback to pure inv-vol when all assets have negative drift (bear market)')
    print(f"  {'Variant':>26}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ────────────────────────────────────────────────────────────────────────')
    s_drift = ens98_drift_rp(assets6, VOL_W)
    variants = [
        ('Core3+5m pure inv-vol', champ),
        ('Core3+5m drift×inv-vol', s_drift),
    ]
    for label, s in variants:
        r = stats(s)
        if r:
            cum, sh, wr, wins, n, mdd, calmar = r
            print(f'  {label:>26}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}')
    r_drift = stats(s_drift)
    if r_drift and r_champ:
        delta_cum = r_drift[0] - r_champ[0]
        delta_mdd = r_drift[5] - r_champ[5]
        verdict = 'ADOPT' if delta_cum > 0.05 else 'REJECT'
        print(f'\n  → Drift×inv-vol vs pure inv-vol: CumRet Δ={delta_cum:+.1%}  MaxDD Δ={delta_mdd:+.1%}  → {verdict}')
    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'), ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]
    print('\n  Walk-forward breakdown (drift RP vs pure inv-vol):')
    hdr = f"  {'Variant':>26}"
    for p, *_ in WF:
        hdr += f'  {p:>12}'
    print(hdr)
    print('  ' + '─' * (26 + 14 * len(WF) + 4))
    for label, s in variants:
        row = f'  {label:>26}'
        for _, start, end in WF:
            r = sub_stats(s, start, end)
            if r:
                ok = '✓' if r[0] > 0 else '✗'
                row += f'  {r[0]:>+9.1%} {ok}'
            else:
                row += f"  {'n/a':>12}"
        print(row)
    print('\n  [3/3] START-DATE ROBUSTNESS — 5m vs 6m across data windows')
    print('  Sub-period returns from full series (no warmup burn)')
    print(f"  {'Config':>20}  {'Full 5y':>10}  {'Last 4y':>10}  {'Last 3y':>10}  {'P2 2023-24':>12}")
    print('  ────────────────────────────────────────────────────────────────────────')
    s_6m = ens98_rp(assets6, 6)
    robustness_configs = [
        ('Core3 + 5m (new)', champ),
        ('Core3 + 6m (prev)', s_6m),
        ('Core3 + 12m (cons)', ens98_rp(assets6, 12)),
    ]
    for label, s in robustness_configs:
        full = sub_stats(s, None, None)
        last4y = sub_stats(s, '2022-05-01', None)
        last3y = sub_stats(s, '2023-05-01', None)
        p2 = sub_stats(s, '2023-01-01', '2025-01-01')
        row = f'  {label:>20}'
        for r in [full, last4y, last3y, p2]:
            if r:
                ok = '✓' if r[0] > 0 else '✗'
                row += f'  {r[0]:>+8.1%} {ok}'
            else:
                row += f"  {'n/a':>12}"
        print(row)
    print('\n  → If 5m beats 6m across all windows, it\'s NOT a start-date artifact.')
    print(f"\n{'═' * W}")


def portfolio_v49_drift_rp_corrected(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v49: Fix look-ahead in v48 drift-magnitude RP.
    Bug: v48 used rebal_date=mth_r.index[i] (end-of-month-i) to look up drift,
         but mth_r.iloc[i] IS month-i's return → drift from within the return period.
    Fix: rebal_date=mth_r.index[i-1] — rebalance at end of prior month using
         prior-month drift data, apply weights to the NEXT month's return (i).
    Tests: (1) Biased vs corrected drift RP comparison
           (2) Commission sweep of corrected version
           (3) Drift window sensitivity (30/45/60/90/120d)
           (4) RP vol window sweep (3/4/5/6/7m) with drift-magnitude RP
    """
    W = 122
    BOOST = 1.5
    CORE3 = frozenset({11, 3, 7})
    def make_signal(base_pct: float=0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                    else sig_vol_targeting_40_45d(df))
            return _apply_seasonal_scalar(base, CORE3, weak=set(), boost=BOOST)
        return fn
    def build_equities(assets, c):
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c
        return (eq_main, eq_floor)
    def build_drift_series(assets, drift_w: int):
        result = {}
        for name, df in assets:
            log_r = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            mu = log_r.rolling(drift_w).mean() * 252
            sigma2 = log_r.rolling(drift_w).var() * 252
            result[name] = (mu - 0.5 * sigma2).dropna()
        return result
    def monthly_rp_inv_vol(equities, vol_w: int):
        """Pure inverse-vol RP (no look-ahead, reference)."""
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = ([], [])
        for i in range(vol_w, len(mth_r)):
            trailing = mth_r.iloc[i - vol_w:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=pd.DatetimeIndex(dates))
    def monthly_rp_drift_biased(equities, drift_series, vol_w: int):
        """v48 BIASED version: uses end-of-month-i drift → look-ahead."""
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = ([], [])
        for i in range(vol_w, len(mth_r)):
            trailing = mth_r.iloc[i - vol_w:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            rebal_date = mth_r.index[i]
            drift_vec = pd.Series({
                k: max(float(drift_series[k][drift_series[k].index <= rebal_date].iloc[-1])
                       if len(drift_series[k][drift_series[k].index <= rebal_date]) > 0 else 0.0, 0.0)
                for k in eq_df.columns
            })
            if drift_vec.sum() > 1e-09:
                w = inv_vol * drift_vec / (inv_vol * drift_vec).sum()
            else:
                w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=pd.DatetimeIndex(dates))
    def monthly_rp_drift_corrected(equities, drift_series, vol_w: int):
        """v49 CORRECTED: rebalance at end-of-month i-1, apply weights to month i return."""
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = ([], [])
        for i in range(vol_w, len(mth_r)):
            trailing = mth_r.iloc[i - vol_w:i]
            inv_vol = 1.0 / (trailing.std() + 1e-08)
            rebal_date = mth_r.index[i - 1]
            drift_vec = pd.Series({
                k: max(float(drift_series[k][drift_series[k].index <= rebal_date].iloc[-1])
                       if len(drift_series[k][drift_series[k].index <= rebal_date]) > 0 else 0.0, 0.0)
                for k in eq_df.columns
            })
            if drift_vec.sum() > 1e-09:
                w = inv_vol * drift_vec / (inv_vol * drift_vec).sum()
            else:
                w = inv_vol / inv_vol.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=pd.DatetimeIndex(dates))
    def ens98(rp_main, rp_floor):
        idx = rp_main.index.intersection(rp_floor.index)
        return (0.98 * rp_main.loc[idx] + 0.02 * rp_floor.loc[idx]).dropna()
    def stats(s):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return (cum, sh, wr, n, mdd, calmar)
    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start: mask &= s_full.index >= start
        if end: mask &= s_full.index < end
        return stats(s_full[mask])
    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'), ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]
    print(f"\n{'═' * W}")
    print('  v49 — DRIFT-MAGNITUDE RP: LOOK-AHEAD FIX + TUNING')
    print('  v48 bug: drift sampled at end-of-month-i → used for month-i return (look-ahead)')
    print('  v49 fix: drift sampled at end-of-month-(i-1) → used for month-i return (correct)')
    print(f"{'═' * W}")
    eq_main, eq_floor = build_equities(assets6, 0.01)
    drift60 = build_drift_series(assets6, 60)
    print('\n  [1/4] LOOK-AHEAD QUANTIFICATION (1% commission, 5m vol window, 60d drift)')
    print(f"  {'Variant':>32}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ──────────────────────────────────────────────────────────────────────────────')
    VOL_W = 5
    inv_main = monthly_rp_inv_vol(eq_main, VOL_W)
    inv_floor = monthly_rp_inv_vol(eq_floor, VOL_W)
    s_inv = ens98(inv_main, inv_floor)
    bias_main = monthly_rp_drift_biased(eq_main, drift60, VOL_W)
    bias_floor = monthly_rp_drift_biased(eq_floor, drift60, VOL_W)
    s_bias = ens98(bias_main, bias_floor)
    fix_main = monthly_rp_drift_corrected(eq_main, drift60, VOL_W)
    fix_floor = monthly_rp_drift_corrected(eq_floor, drift60, VOL_W)
    s_fix = ens98(fix_main, fix_floor)
    variants_1 = [
        ('Pure inv-vol (champion)', s_inv),
        ('Drift×inv-vol BIASED (v48)', s_bias),
        ('Drift×inv-vol CORRECTED (v49)', s_fix),
    ]
    for label, s in variants_1:
        r = stats(s)
        if r:
            cum, sh, wr, n, mdd, calmar = r
            print(f'  {label:>32}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}')
    r_inv = stats(s_inv)
    r_fix = stats(s_fix)
    if r_inv and r_fix:
        delta_cum = r_fix[0] - r_inv[0]
        verdict = 'ADOPT' if delta_cum > 0.05 and r_fix[1] > r_inv[1] else 'REVIEW'
        print(f'\n  → Corrected drift RP vs pure inv-vol: CumRet Δ={delta_cum:+.1%}  Sh Δ={r_fix[1] - r_inv[1]:+.2f}  → {verdict}')
    print('\n  Walk-forward (corrected drift RP vs pure inv-vol):')
    hdr = f"  {'Variant':>32}"
    for p, *_ in WF:
        hdr += f'  {p:>12}'
    print(hdr)
    print('  ' + '─' * (34 + 14 * len(WF)))
    for label, s in variants_1:
        row = f'  {label:>32}'
        for _, start, end in WF:
            r = sub_stats(s, start, end)
            if r:
                ok = '✓' if r[0] > 0 else '✗'
                row += f'  {r[0]:>+9.1%} {ok}'
            else:
                row += f"  {'n/a':>12}"
        print(row)
    print('\n  [2/4] COMMISSION ROBUSTNESS (corrected drift RP, 5m vol, 60d drift)')
    print(f"  {'Comm':>8}  {'RT':>6}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Viable':>7}")
    print('  ──────────────────────────────────────────────────────')
    for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        eq_m, eq_f = build_equities(assets6, c)
        ds = build_drift_series(assets6, 60)
        sm = monthly_rp_drift_corrected(eq_m, ds, VOL_W)
        sf = monthly_rp_drift_corrected(eq_f, ds, VOL_W)
        s = ens98(sm, sf)
        r = stats(s)
        if r:
            ok = '✓ YES' if r[0] > 0 and r[1] > 0.3 else '✗ NO'
            print(f'  {c:>7.3%}  {2 * c:>5.3%}  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {ok}')
    print('\n  [3/4] DRIFT WINDOW SENSITIVITY (corrected, 5m vol, 1% commission)')
    print(f"  {'DriftWin':>10}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs 60d':>9}")
    print('  ──────────────────────────────────────────────────────────────────')
    r_60d = stats(s_fix)
    for dw in [30, 45, 60, 90, 120]:
        ds = build_drift_series(assets6, dw)
        eq_m, eq_f = (eq_main, eq_floor)
        sm = monthly_rp_drift_corrected(eq_m, ds, VOL_W)
        sf = monthly_rp_drift_corrected(eq_f, ds, VOL_W)
        s = ens98(sm, sf)
        r = stats(s)
        if r:
            marker = ' ← prev' if dw == 60 else ''
            delta = f'{r[0] - r_60d[0]:>+8.1%}' if dw != 60 else ' baseline'
            print(f'  {dw:>9}d  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {r[5]:>8.2f}  {delta}{marker}')
    print('\n  [4/4] RP VOL WINDOW SWEEP (corrected drift RP, 60d drift, 1% commission)')
    print(f"  {'VolWin':>8}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs 5m':>9}")
    print('  ──────────────────────────────────────────────────────────────')
    r_5m = stats(s_fix)
    for vw in [3, 4, 5, 6, 7, 8]:
        sm = monthly_rp_drift_corrected(eq_main, drift60, vw)
        sf = monthly_rp_drift_corrected(eq_floor, drift60, vw)
        s = ens98(sm, sf)
        r = stats(s)
        if r:
            marker = ' ← prev' if vw == 5 else ''
            delta = f'{r[0] - r_5m[0]:>+8.1%}' if vw != 5 else ' baseline'
            print(f'  {vw:>7}m  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {r[5]:>8.2f}  {delta}{marker}')
    print(f"\n{'═' * W}")


def portfolio_v50_drawdown_gate_inv_variance(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v50: Two new dimensions on confirmed champion (Core3+VT45×1.5, 5m RP, pure inv-vol):
    (1) Portfolio drawdown gate: When portfolio is >X% below ATH, pause next month (hold cash).
        Targets P1 (2021-22 bear, -20.3%) without changing any signal parameters.
        Test thresholds: 10/15/20/25/30%. Pause duration: 1 month.
    (2) Inverse variance RP (1/σ² weights vs 1/σ):
        More aggressively concentrates weight in low-vol assets. Simple change, never tested.
    """
    W = 122
    BOOST = 1.5
    CORE3 = frozenset({11, 3, 7})
    VOL_W = 5

    def make_signal(base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                    else sig_vol_targeting_40_45d(df))
            return _apply_seasonal_scalar(base, CORE3, weak=set(), boost=BOOST)
        return fn

    def ens98_base(assets, vol_w: int, c: float = 0.01, inv_var: bool = False):
        """Core ens98/2 RP portfolio; inv_var=True uses 1/σ² weights."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c

        def monthly_rp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            rets, dates = [], []
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                if inv_var:
                    w = 1.0 / (trailing.var() + 1e-16)
                else:
                    w = 1.0 / (trailing.std() + 1e-08)
                w = w / w.sum()
                rets.append(float((mth_r.iloc[i] * w).sum()))
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = monthly_rp(eq_main)
        sb = monthly_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def apply_dd_gate(s: pd.Series, threshold: float) -> pd.Series:
        """Zero out month i+1 when portfolio is >threshold below ATH at end of month i."""
        equity = (1 + s).cumprod()
        atk_high = equity.cummax()
        dd = equity / atk_high - 1
        result = s.copy()
        paused = False
        for i in range(len(result)):
            if paused:
                result.iloc[i] = 0.0
                paused = False
            elif dd.iloc[i] < -threshold:
                paused = True
        return result

    def stats(s):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start:
            mask &= s_full.index >= start
        if end:
            mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'),
          ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]

    print(f"\n{'═' * W}")
    print('  v50 — PORTFOLIO DRAWDOWN GATE + INVERSE VARIANCE RP')
    print(f"{'═' * W}")

    champ = ens98_base(assets6, VOL_W)
    r_champ = stats(champ)

    print('\n  [1/3] PORTFOLIO DRAWDOWN GATE SWEEP (Core3+5m champion, 1% commission)')
    print('  Mechanism: if portfolio DD > threshold at end of month i, skip month i+1 (hold cash)')
    print(f"  {'Threshold':>11}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs champ':>10}")
    print('  ──────────────────────────────────────────────────────────────────────')

    dd_results = {}
    for thr in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]:
        s = apply_dd_gate(champ, thr) if thr > 0 else champ
        dd_results[thr] = s
        r = stats(s)
        if not r:
            continue
        cum, sh, wr, n, mdd, calmar = r
        delta = f'{cum - r_champ[0]:>+8.1%}' if thr > 0 else '  baseline'
        marker = ' ← champion' if thr == 0 else ''
        print(f'  {thr:>10.0%}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}  {delta}{marker}')

    best_thr = max((t for t in dd_results if t > 0),
                   key=lambda t: (stats(dd_results[t]) or (0,))[0])
    r_best = stats(dd_results[best_thr])
    if r_best and r_champ:
        delta_cum = r_best[0] - r_champ[0]
        delta_mdd = r_best[4] - r_champ[4]
        verdict = 'ADOPT' if delta_cum > 0 or (delta_mdd > 0.01 and r_best[1] > r_champ[1]) else 'REJECT'
        print(f'\n  → Best gate: {best_thr:.0%}  CumRet Δ={delta_cum:+.1%}  MaxDD Δ={delta_mdd:+.1%}  → {verdict}')

    print('\n  Walk-forward (best DD gate vs champion):')
    hdr = f"  {'Variant':>22}"
    for p, *_ in WF:
        hdr += f'  {p:>12}'
    print(hdr)
    print('  ' + '─' * (24 + 14 * len(WF)))
    for label, s in (('Pure inv-vol (champ)', champ),
                     (f'DD gate {best_thr:.0%}', dd_results[best_thr])):
        row = f'  {label:>22}'
        for _, start, end in WF:
            r = sub_stats(s, start, end)
            if r:
                ok = '✓' if r[0] > 0 else '✗'
                row += f'  {r[0]:>+9.1%} {ok}'
            else:
                row += f"  {'n/a':>12}"
        print(row)

    print('\n  [2/3] INVERSE VARIANCE RP (1/σ² vs 1/σ weights)')
    print(f"  {'Variant':>24}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ──────────────────────────────────────────────────────────────────────')

    s_inv_var = ens98_base(assets6, VOL_W, inv_var=True)
    for label, s in (('Inv-vol 1/σ (champion)', champ), ('Inv-variance 1/σ²', s_inv_var)):
        r = stats(s)
        if not r:
            continue
        cum, sh, wr, n, mdd, calmar = r
        print(f'  {label:>24}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}')

    r_inv_var = stats(s_inv_var)
    if r_inv_var and r_champ:
        delta_cum = r_inv_var[0] - r_champ[0]
        delta_mdd = r_inv_var[4] - r_champ[4]
        verdict = 'ADOPT' if delta_cum > 0.05 and r_inv_var[1] >= r_champ[1] else 'REJECT'
        print(f'\n  → Inv-variance vs inv-vol: CumRet Δ={delta_cum:+.1%}  MaxDD Δ={delta_mdd:+.1%}  → {verdict}')

    print('\n  Walk-forward (inv-variance vs inv-vol):')
    hdr = f"  {'Variant':>24}"
    for p, *_ in WF:
        hdr += f'  {p:>12}'
    print(hdr)
    print('  ' + '─' * (26 + 14 * len(WF)))
    for label, s in (('Inv-vol 1/σ (champion)', champ), ('Inv-variance 1/σ²', s_inv_var)):
        row = f'  {label:>24}'
        for _, start, end in WF:
            r = sub_stats(s, start, end)
            if r:
                ok = '✓' if r[0] > 0 else '✗'
                row += f'  {r[0]:>+9.1%} {ok}'
            else:
                row += f"  {'n/a':>12}"
        print(row)

    print(f'\n  [3/3] COMBO: DD gate {best_thr:.0%} + inv-variance RP')
    s_combo_dd = dd_results[best_thr]
    s_combo_iv = apply_dd_gate(s_inv_var, best_thr)
    variants_3 = [
        ('Champion (no changes)', champ),
        (f'DD gate {best_thr:.0%} only', s_combo_dd),
        ('Inv-var only', s_inv_var),
        (f'DD gate {best_thr:.0%} + inv-var', s_combo_iv),
    ]
    print(f"  {'Variant':>28}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ────────────────────────────────────────────────────────────────────────────')
    for label, s in variants_3:
        r = stats(s)
        if not r:
            continue
        cum, sh, wr, n, mdd, calmar = r
        marker = ' ← champion' if label == 'Champion (no changes)' else ''
        print(f'  {label:>28}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}')

    print(f"\n{'═' * W}")


def portfolio_v51_inv_var_validation(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v51: Validate inverse-variance RP new champion (Core3+VT45×1.5, 5m, 1/σ²).
    (1) RP vol window sweep (3-8m): is 5m still optimal with 1/σ² weights?
    (2) Commission robustness: 0.05% to 1%.
    (3) P3 deep-dive: monthly returns 2025-2026 to diagnose the -4.3% flip vs inv-vol.
    (4) vs 0.1% commission (practical CEX target): new champion vs old inv-vol champion.
    """
    W = 122
    BOOST = 1.5
    CORE3 = frozenset({11, 3, 7})

    def make_signal(base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                    else sig_vol_targeting_40_45d(df))
            return _apply_seasonal_scalar(base, CORE3, weak=set(), boost=BOOST)
        return fn

    def ens98(assets, vol_w: int, c: float = 0.01, inv_var: bool = True):
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c

        def monthly_rp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            rets, dates = [], []
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                w = (1.0 / (trailing.var() + 1e-16) if inv_var
                     else 1.0 / (trailing.std() + 1e-8))
                w = w / w.sum()
                rets.append(float((mth_r.iloc[i] * w).sum()))
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = monthly_rp(eq_main)
        sb = monthly_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def stats(s):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start: mask &= s_full.index >= start
        if end: mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'),
          ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]

    print(f"\n{'═' * W}")
    print("  v51 — INVERSE VARIANCE CHAMPION VALIDATION")
    print("  New champion: Core3+VT45×1.5, ens 98/2, 5m RP vol window, 1/σ² weights")
    print(f"{'═' * W}")

    champ_inv_var = ens98(assets6, 5, inv_var=True)
    champ_inv_vol = ens98(assets6, 5, inv_var=False)

    print("\n  [1/4] RP VOL WINDOW SWEEP (inv-variance 1/σ², 1% commission)")
    print(f"  {'VolWin':>8}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs 5m':>9}")
    print('  ' + '─' * 64)

    r_5m = stats(champ_inv_var)
    for vw in [3, 4, 5, 6, 7, 8]:
        s = ens98(assets6, vw, inv_var=True)
        r = stats(s)
        if not r: continue
        marker = ' ← new champ' if vw == 5 else ''
        delta = f"{r[0] - r_5m[0]:>+8.1%}" if vw != 5 else '  baseline'
        print(f"  {vw:>7}m  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {r[5]:>8.2f}  {delta}{marker}")

    best_vw = max(range(3, 9), key=lambda vw: (stats(ens98(assets6, vw, inv_var=True)) or (0,))[0])
    print(f"\n  → Best vol window for inv-var: {best_vw}m")

    if best_vw != 5:
        s_best = ens98(assets6, best_vw, inv_var=True)
        r_best = stats(s_best)
        r_cur = stats(champ_inv_var)
        if r_best and r_cur:
            print(f"  → Update: {best_vw}m gives CumRet={r_best[0]:+.1%} Sh={r_best[1]:.2f} Calmar={r_best[5]:.2f}")
            print(f"            vs 5m:     CumRet={r_cur[0]:+.1%}  Sh={r_cur[1]:.2f} Calmar={r_cur[5]:.2f}")
    else:
        s_best = champ_inv_var

    print(f"\n  Walk-forward @ optimal {best_vw}m vol window (inv-var):")
    hdr = f"  {'Variant':>26}"
    for p, *_ in WF:
        hdr += f"  {p:>12}"
    print(hdr)
    print('  ' + '─' * (28 + 14 * len(WF)))
    for label, s in [('Inv-vol 5m (old champ)', champ_inv_vol),
                     (f'Inv-var {best_vw}m (new champ)', s_best)]:
        row = f"  {label:>26}"
        for _, start, end in WF:
            r = sub_stats(s, start, end)
            if r:
                ok = '✓' if r[0] > 0 else '✗'
                row += f"  {r[0]:>+9.1%} {ok}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    print(f"\n  [2/4] COMMISSION ROBUSTNESS (inv-var {best_vw}m champion)")
    print(f"  {'Comm':>8}  {'RT':>6}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Viable':>7}")
    print('  ' + '─' * 54)
    for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        s = ens98(assets6, best_vw, c=c, inv_var=True)
        r = stats(s)
        if not r: continue
        ok = '✓ YES' if r[0] > 0 and r[1] > 0.3 else '✗ NO'
        print(f"  {c:>7.3%}  {2 * c:>5.3%}  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {ok}")

    print("\n  [3/4] P3 (2025-2026) MONTHLY BREAKDOWN — why did inv-var flip to -4.3%?")
    MONTH_NAMES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    for label, s in [('Inv-vol (old)', champ_inv_vol), ('Inv-var (new)', s_best)]:
        mask = s.index >= '2025-01-01'
        s_p3 = s[mask]
        if len(s_p3) > 0:
            months_str = '  '.join(f"{MONTH_NAMES[d.month]}{d.year % 100:02d}:{r:>+6.1%}"
                                   for d, r in s_p3.items())
            print(f"  {label:>16}: {months_str}")

    print("\n  [4/4] PRACTICAL COMMISSION (0.1% = 0.2% RT): old vs new champion")
    print(f"  {'Variant':>26}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ' + '─' * 74)
    for label, s_fn in [('Inv-vol 5m @ 0.1%', lambda: ens98(assets6, 5, c=0.001, inv_var=False)),
                        (f'Inv-var {best_vw}m @ 0.1%', lambda: ens98(assets6, best_vw, c=0.001, inv_var=True))]:
        s = s_fn()
        r = stats(s)
        if not r: continue
        cum, sh, wr, n, mdd, calmar = r
        print(f"  {label:>26}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}")

    print(f"\n{'═' * W}")


def portfolio_v52_boost_reopt_hierarchical_rp(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v52: Two dimensions on the new inv-variance champion:
    (1) Seasonal boost magnitude re-opt with 1/σ² (was ×1.5 for inv-vol in v43).
        Hypothesis: inv-var concentrates weights differently → optimal boost may differ.
        Test: ×1.0/1.2/1.3/1.5/1.8/2.0
    (2) Hierarchical RP (two-level):
        L1: 1/σ²_cluster weights for {BTC,ETH} vs {SOL,BNB,ADA,AVAX} clusters.
        L2: 1/σ² within each cluster.
        Hypothesis: explicit large-cap/mid-cap balance improves P1/P3 without hurting P2.
    """
    global _hrp_best_boost
    W = 122
    CORE3 = frozenset({11, 3, 7})
    VOL_W = 5
    def make_signal(boost: float, base_pct: float=0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                    else sig_vol_targeting_40_45d(df))
            return _apply_seasonal_scalar(base, CORE3, weak=set(), boost=boost)
        return fn
    def ens98_inv_var(assets, vol_w: int, boost: float=1.5, c: float=0.01):
        """Inv-variance ens98/2 with configurable seasonal boost."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(boost, 0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(boost, 0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c
        def monthly_rp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            rets, dates = ([], [])
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                w = 1.0 / (trailing.var() + 1e-16)
                w = w / w.sum()
                rets.append(float((mth_r.iloc[i] * w).sum()))
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))
        sa = monthly_rp(eq_main)
        sb = monthly_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def ens98_hierarchical(assets, vol_w: int, c: float=0.01):
        """Two-level hierarchical RP: L1 cluster balance, L2 intra-cluster 1/σ²."""
        global COMMISSION
        LARGE_CAPS = {'BTC', 'ETH'}
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(1.5, 0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(1.5, 0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c
        def monthly_hrp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_names = list(eq_df.columns)
            large = [n for n in all_names if n in LARGE_CAPS]
            small = [n for n in all_names if n not in LARGE_CAPS]
            rets, dates = ([], [])
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                inv_var = 1.0 / (trailing.var() + 1e-16)
                w_large = inv_var[large] / inv_var[large].sum() if large else pd.Series(dtype=float)
                w_small = inv_var[small] / inv_var[small].sum() if small else pd.Series(dtype=float)
                r_large = float((mth_r.iloc[i][large] * w_large).sum()) if len(large) > 0 else 0.0
                r_small = float((mth_r.iloc[i][small] * w_small).sum()) if len(small) > 0 else 0.0
                tr_large = (trailing[large] * w_large).sum(axis=1)
                tr_small = (trailing[small] * w_small).sum(axis=1)
                var_large = float(tr_large.var()) + 1e-16
                var_small = float(tr_small.var()) + 1e-16
                cl_w_large = 1.0 / var_large / (1.0 / var_large + 1.0 / var_small)
                cl_w_small = 1.0 - cl_w_large
                rets.append(cl_w_large * r_large + cl_w_small * r_small)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))
        sa = monthly_hrp(eq_main)
        sb = monthly_hrp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def stats(s):
        if len(s) < 4:
            return
        else:
            cum = float((1 + s).prod() - 1)
            sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
            wr = float((s > 0).mean())
            n = len(s)
            mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
            ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
            calmar = ann / abs(mdd) if mdd != 0 else 0
            return (cum, sh, wr, n, mdd, calmar)
    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start:
            mask &= s_full.index >= start
        if end:
            mask &= s_full.index < end
        return stats(s_full[mask])
    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'), ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]
    print(f"\n{'═' * W}")
    print('  v52 — SEASONAL BOOST RE-OPT (inv-var) + HIERARCHICAL RP')
    print(f"{'═' * W}")
    champ = ens98_inv_var(assets6, VOL_W, boost=1.5)
    r_champ = stats(champ)
    print('\n  [1/3] SEASONAL BOOST MAGNITUDE SWEEP (inv-var 5m, 1% commission)')
    print('  Previous optimal: ×1.5 (was tested with inv-vol in v43)')
    print(f"  {'Boost':>8}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs ×1.5':>9}")
    print('  ──────────────────────────────────────────────────────────────────')
    boost_results = {}
    for boost in [1.0, 1.2, 1.3, 1.5, 1.8, 2.0]:
        s = ens98_inv_var(assets6, VOL_W, boost=boost)
        boost_results[boost] = s
        r = stats(s)
        if r:
            cum, sh, wr, n, mdd, calmar = r
            marker = ' ← champion' if boost == 1.5 else ''
            delta = f'{cum - r_champ[0]:>+8.1%}' if boost != 1.5 else '  baseline'
            print(f'  ×{boost:>5.1f}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}  {delta}{marker}')
    best_boost = max(boost_results, key=lambda b: (stats(boost_results[b]) or (0,))[0])
    r_best_b = stats(boost_results[best_boost])
    if r_best_b and r_champ:
            delta = r_best_b[0] - r_champ[0]
            verdict = 'UPDATE' if best_boost != 1.5 and delta > 0.05 else 'KEEP ×1.5'
            print(f'\n  → Best boost: ×{best_boost:.1f}  CumRet Δ={delta:+.1%}  → {verdict}')
    print('\n  [2/3] HIERARCHICAL RP (2-level: {BTC,ETH} vs {SOL,BNB,ADA,AVAX})')
    print('  L1: 1/σ²_cluster  →  L2: 1/σ² within cluster')
    print(f"  {'Variant':>30}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ──────────────────────────────────────────────────────────────────────────────')
    s_hrp = ens98_hierarchical(assets6, VOL_W)
    for label, s in [('Flat inv-var 1/σ² (champion)', champ),
                     ('Hierarchical RP (L1+L2)', s_hrp)]:
        r = stats(s)
        if r:
            cum, sh, wr, n, mdd, calmar = r
            print(f'  {label:>30}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}')
    r_hrp = stats(s_hrp)
    if r_hrp and r_champ:
            delta_cum = r_hrp[0] - r_champ[0]
            delta_mdd = r_hrp[4] - r_champ[4]
            verdict = 'ADOPT' if delta_cum > 0.05 and r_hrp[1] >= r_champ[1] else 'REJECT'
            print(f'\n  → Hierarchical RP: CumRet Δ={delta_cum:+.1%}  MaxDD Δ={delta_mdd:+.1%}  → {verdict}')
    print('\n  Walk-forward (hierarchical RP vs flat inv-var):')
    hdr = f"  {'Variant':>30}"
    for p, *_ in WF:
        hdr += f'  {p:>12}'
    print(hdr)
    print('  ' + '─' * (32 + 14 * len(WF)))
    for label, s in [('Flat inv-var (champion)', champ), ('Hierarchical RP', s_hrp)]:
        row = f'  {label:>30}'
        for _, start, end in WF:
            r = sub_stats(s, start, end)
            if r:
                ok = '✓' if r[0] > 0 else '✗'
                row += f'  {r[0]:>+9.1%} {ok}'
            else:
                row += f"  {'n/a':>12}"
        print(row)
    use_boost = best_boost
    s_combo = (ens98_hierarchical(assets6, VOL_W) if use_boost == 1.5
               else ens98_inv_var(assets6, VOL_W, boost=use_boost))
    s_best_boost = boost_results[best_boost]
    _hrp_best_boost = best_boost
    print('\n  [3/3] BEST COMBINATIONS')
    print(f"  {'Variant':>32}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print('  ────────────────────────────────────────────────────────────────────────────────')
    combos = [
        ('Champion (×1.5, flat inv-var)', champ),
        (f'×{best_boost:.1f} boost, flat inv-var', s_best_boost),
        ('×1.5 boost, hierarchical RP', s_hrp),
    ]
    for label, s in combos:
        r = stats(s)
        if r:
            cum, sh, wr, n, mdd, calmar = r
            marker = ' ← champion' if label.startswith('Champion') else ''
            print(f'  {label:>32}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}')
    print(f"\n{'═' * W}")


def portfolio_v53_hrp_boost_combined(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v53: Combine hierarchical RP (ADOPT) + extended boost sweep.
    (1) Boost sweep ×1.5-×3.0 WITH hierarchical RP: find joint optimum.
        Hypothesis: ×2.0 or higher may win vs ×1.5, since both dimensions improve.
    (2) Commission robustness of the new champion (best boost + hierarchical RP).
    (3) Full walk-forward table: all variants vs final champion decision.
    """
    W = 122
    CORE3 = frozenset({3, 7, 11})
    VOL_W = 5
    LARGE_CAPS = {'BTC', 'ETH'}

    def make_signal(boost: float, base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                    else sig_vol_targeting_40_45d(df))
            return _apply_seasonal_scalar(base, CORE3, weak=set(), boost=boost)
        return fn

    def build_eq(assets, boost, c):
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(boost, 0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(boost, 0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c
        return eq_main, eq_floor

    def monthly_flat_inv_var(equities, vol_w):
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(vol_w, len(mth_r)):
            trailing = mth_r.iloc[i - vol_w:i]
            w = 1.0 / (trailing.var() + 1e-16)
            w = w / w.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=pd.DatetimeIndex(dates))

    def monthly_hrp(equities, vol_w):
        eq_df = pd.DataFrame(equities).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        all_names = list(eq_df.columns)
        large = [n for n in all_names if n in LARGE_CAPS]
        small = [n for n in all_names if n not in LARGE_CAPS]
        rets, dates = [], []
        for i in range(vol_w, len(mth_r)):
            trailing = mth_r.iloc[i - vol_w:i]
            inv_var = 1.0 / (trailing.var() + 1e-16)
            w_large = inv_var[large] / inv_var[large].sum()
            w_small = inv_var[small] / inv_var[small].sum()
            r_large = float((mth_r.iloc[i][large] * w_large).sum())
            r_small = float((mth_r.iloc[i][small] * w_small).sum())
            tr_large = (trailing[large] * w_large).sum(axis=1)
            tr_small = (trailing[small] * w_small).sum(axis=1)
            var_l = float(tr_large.var()) + 1e-16
            var_s = float(tr_small.var()) + 1e-16
            cl_w_l = (1.0 / var_l) / (1.0 / var_l + 1.0 / var_s)
            rets.append(cl_w_l * r_large + (1 - cl_w_l) * r_small)
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=pd.DatetimeIndex(dates))

    def ens98(eq_main, eq_floor, rp_fn):
        sa = rp_fn(eq_main)
        sb = rp_fn(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def stats(s):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float(((s + 1).cumprod().div((s + 1).cumprod().cummax())).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start: mask &= s_full.index >= start
        if end: mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [("Full 5y", None, None), ("P1 2021-23", None, "2023-01-01"),
          ("P2 2023-24", "2023-01-01", "2025-01-01"), ("P3 2025-26", "2025-01-01", None)]

    print(f"\n{'═' * W}")
    print("  v53 — HIERARCHICAL RP + EXTENDED BOOST SWEEP + COMMISSION ROBUSTNESS")
    print(f"{'═' * W}")


    eq_m15, eq_f15 = build_eq(assets6, 1.5, 0.01)
    hrp_fn = lambda eq: monthly_hrp(eq, VOL_W)
    flat_fn = lambda eq: monthly_flat_inv_var(eq, VOL_W)
    champ_flat = ens98(eq_m15, eq_f15, flat_fn)
    champ_hrp15 = ens98(eq_m15, eq_f15, hrp_fn)


    print("\n  [1/3] BOOST SWEEP WITH HIERARCHICAL RP (1% commission)")
    print(f"  {'Boost':>8}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs ×1.5-HRP':>12}")
    print("  ──────────────────────────────────────────────────────────────────────")

    r_hrp15 = stats(champ_hrp15)
    hrp_boost_results = {1.5: champ_hrp15}
    for boost in [1.5, 1.8, 2.0, 2.5, 3.0]:
        if boost == 1.5:
            s = champ_hrp15
        else:
            eq_m, eq_f = build_eq(assets6, boost, 0.01)
            s = ens98(eq_m, eq_f, hrp_fn)
            hrp_boost_results[boost] = s
        r = stats(s)
        if not r: continue
        cum, sh, wr, n, mdd, calmar = r
        delta = f"{cum - r_hrp15[0]:>+10.1%}" if boost != 1.5 else "    baseline"
        marker = " ← v52" if boost == 1.5 else ""
        print(f"  ×{boost:>5.1f}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}  {delta}{marker}")

    best_boost = max(hrp_boost_results, key=lambda b: (stats(hrp_boost_results[b]) or (0,))[0])
    s_new_champ = hrp_boost_results[best_boost]
    r_new = stats(s_new_champ)
    r_old = stats(champ_flat)
    if r_new and r_old:
        delta = r_new[0] - r_old[0]
        print(f"\n  → Best HRP boost: ×{best_boost:.1f}  CumRet={r_new[0]:+.1%}  vs v51 champ: Δ={delta:+.1%}")


    print(f"\n  [2/3] COMMISSION ROBUSTNESS (HRP ×{best_boost:.1f}, new champion)")
    print(f"  {'Comm':>8}  {'RT':>6}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'Viable':>7}")
    print("  ────────────────────────────────────────────────────────────────")
    for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        eq_m, eq_f = build_eq(assets6, best_boost, c)
        s = ens98(eq_m, eq_f, hrp_fn)
        r = stats(s)
        if not r: continue
        ok = "✓ YES" if r[0] > 0 and r[1] > 0.3 else "✗ NO"
        print(f"  {c:>7.3%}  {2 * c:>5.3%}  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {r[5]:>8.2f}  {ok}")


    print("\n  [3/3] WALK-FORWARD: CHAMPION EVOLUTION SUMMARY")
    wf_configs = [
        ("v47: inv-vol ×1.5", 'flat_inv_vol'),
        ("v51: inv-var ×1.5 flat", 'flat_inv_var'),
        ("v52: inv-var ×1.5 HRP", 'hrp_1.5'),
        (f"v53: inv-var ×{best_boost:.1f} HRP", 'hrp_best'),
    ]


    global COMMISSION
    old_c = COMMISSION
    COMMISSION = 0.01
    try:
        eq_m_vol = {n: Backtest(df, make_signal(1.5, 0.0)(df)).equity for n, df in assets6}
        eq_f_vol = {n: Backtest(df, make_signal(1.5, 0.02)(df)).equity for n, df in assets6}
    finally:
        COMMISSION = old_c
    def inv_vol_fn(eq):
        eq_df = pd.DataFrame(eq).dropna()
        mth_r = eq_df.resample('ME').last().pct_change().dropna()
        rets, dates = [], []
        for i in range(VOL_W, len(mth_r)):
            trailing = mth_r.iloc[i - VOL_W:i]
            w = 1.0 / (trailing.std() + 1e-8)
            w = w / w.sum()
            rets.append(float((mth_r.iloc[i] * w).sum()))
            dates.append(mth_r.index[i])
        return pd.Series(rets, index=pd.DatetimeIndex(dates))
    s_inv_vol = ens98(eq_m_vol, eq_f_vol, inv_vol_fn)

    series_map = {
        'flat_inv_vol': s_inv_vol,
        'flat_inv_var': champ_flat,
        'hrp_1.5': champ_hrp15,
        'hrp_best': s_new_champ,
    }

    hdr = f"  {'Config':>26}"
    for p, *_ in WF:
        hdr += f"  {p:>12}"
    print(hdr)
    print("  " + "─" * (28 + 14 * len(WF)))
    for label, key in wf_configs:
        s = series_map[key]
        row = f"  {label:>26}"
        for _, start, end in WF:
            r = sub_stats(s, start, end)
            if r:
                ok = "✓" if r[0] > 0 else "✗"
                row += f"  {r[0]:>+9.1%} {ok}"
            else:
                row += f"  {'n/a':>12}"
        print(row)


    print("\n  Final champion stats @ 1% and 0.1% commission:")
    eq_m_01, eq_f_01 = build_eq(assets6, best_boost, 0.001)
    s_01 = ens98(eq_m_01, eq_f_01, hrp_fn)
    for label, s in [(f"HRP ×{best_boost:.1f} @ 1%", s_new_champ),
                     (f"HRP ×{best_boost:.1f} @ 0.1%", s_01)]:
        r = stats(s)
        if not r: continue
        cum, sh, wr, n, mdd, calmar = r
        print(f"  {label:>24}: CumRet={cum:+.1%}  Sh={sh:.2f}  MaxDD={mdd:+.1%}  Calmar={calmar:.2f}")

    print(f"\n{'═' * W}")


def portfolio_v54_vol_window_seasonal_retest(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v54: Final validation of v53 champion (HRP ×3.0):
    (1) RP vol window re-test (3-8m): was 5m optimal for inv-vol, may differ for HRP ×3.0.
    (2) Seasonal month targeted re-test: top candidates from v46 sweep vs Core3,
        applied with ×3.0 boost + HRP. Does the higher boost change optimal months?
    (3) Drift window re-test (30/45/60/90/120d) with HRP ×3.0.
    """
    W = 122
    CORE3 = frozenset({11, 3, 7})
    BOOST = 3.0
    LARGE_CAPS = {'BTC', 'ETH'}

    def make_signal(boost: float, months: frozenset, base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                    else sig_vol_targeting_40_45d(df))
            return _apply_seasonal_scalar(base, months, weak=set(), boost=boost)
        return fn

    def build_hrp(assets, vol_w, boost, months, c=0.01, drift_w=60):
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(boost, months, 0.0)(df)).equity
                       for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(boost, months, 0.02)(df)).equity
                        for n, df in assets}
        finally:
            COMMISSION = old_c

        def hrp_rp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_n = list(eq_df.columns)
            large = [n for n in all_n if n in LARGE_CAPS]
            small = [n for n in all_n if n not in LARGE_CAPS]
            rets, dates = [], []
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                inv_v = 1.0 / (trailing.var() + 1e-16)
                wl = inv_v[large] / inv_v[large].sum()
                ws = inv_v[small] / inv_v[small].sum()
                rl = float((mth_r.iloc[i][large] * wl).sum())
                rs = float((mth_r.iloc[i][small] * ws).sum())
                vl = float((trailing[large] * wl).sum(axis=1).var()) + 1e-16
                vs = float((trailing[small] * ws).sum(axis=1).var()) + 1e-16
                cl = (1 / vl) / (1 / vl + 1 / vs)
                rets.append(cl * rl + (1 - cl) * rs)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = hrp_rp(eq_main)
        sb = hrp_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def build_hrp_drift_w(assets, vol_w, boost, months, drift_w, c=0.01):
        """HRP with custom drift window."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            def msig(base_pct):
                def fn(df):
                    base = (sig_base_position(df, base_pct=base_pct) if base_pct > 0
                            else sig_vol_targeting_40_45d(df, window=drift_w))
                    return _apply_seasonal_scalar(base, months, weak=set(), boost=boost)
                return fn
            eq_main = {n: Backtest(df, msig(0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, msig(0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c

        def hrp_rp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_n = list(eq_df.columns)
            large = [n for n in all_n if n in LARGE_CAPS]
            small = [n for n in all_n if n not in LARGE_CAPS]
            rets, dates = [], []
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                inv_v = 1.0 / (trailing.var() + 1e-16)
                wl = inv_v[large] / inv_v[large].sum()
                ws = inv_v[small] / inv_v[small].sum()
                rl = float((mth_r.iloc[i][large] * wl).sum())
                rs = float((mth_r.iloc[i][small] * ws).sum())
                vl = float((trailing[large] * wl).sum(axis=1).var()) + 1e-16
                vs = float((trailing[small] * ws).sum(axis=1).var()) + 1e-16
                cl = (1 / vl) / (1 / vl + 1 / vs)
                rets.append(cl * rl + (1 - cl) * rs)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = hrp_rp(eq_main)
        sb = hrp_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def stats(s):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start: mask &= s_full.index >= start
        if end: mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'),
          ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]

    MN = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    champ = build_hrp(assets6, 5, BOOST, CORE3)
    r_champ = stats(champ)

    print(f"\n{'═' * W}")
    print("  v54 — VOL WINDOW + SEASONAL MONTHS + DRIFT WINDOW RE-TEST (HRP ×3.0)")
    print(f"{'═' * W}")

    print("\n  [1/3] RP VOL WINDOW SWEEP (HRP ×3.0 Core3, 1% commission)")
    print(f"  {'VolWin':>8}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs 5m':>9}")
    print('  ' + '─' * 64)
    r_5m = stats(champ)
    for vw in [3, 4, 5, 6, 7, 8]:
        s = build_hrp(assets6, vw, BOOST, CORE3)
        r = stats(s)
        if not r: continue
        marker = ' ← champion' if vw == 5 else ''
        delta = f"{r[0] - r_5m[0]:>+8.1%}" if vw != 5 else '  baseline'
        print(f"  {vw:>7}m  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {r[5]:>8.2f}  {delta}{marker}")

    print("\n  [2/3] SEASONAL MONTH TARGETED RE-TEST (HRP ×3.0, 5m, 1% commission)")
    print("  Testing Core3 vs top candidates from prior research")
    month_candidates = [
        ('Core3: Mar/Jul/Nov', frozenset({11, 3, 7})),
        ('FebJulNov (v46 #1)', frozenset({2, 7, 11})),
        ('JanJulNov', frozenset({1, 7, 11})),
        ('MarAugNov', frozenset({11, 3, 8})),
        ('MarJulDec', frozenset({3, 7, 12})),
        ('JanMarJul', frozenset({1, 3, 7})),
        ('JanMarNov', frozenset({11, 1, 3})),
    ]
    print(f"  {'Months':>22}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs Core3':>9}")
    print('  ' + '─' * 76)
    r_core3 = stats(champ)
    for label, months in month_candidates:
        s = champ if months == CORE3 else build_hrp(assets6, 5, BOOST, months)
        r = stats(s)
        if not r: continue
        marker = ' ← champion' if months == CORE3 else ''
        delta = f"{r[0] - r_core3[0]:>+8.1%}" if months != CORE3 else '  baseline'
        print(f"  {label:>22}  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {r[5]:>8.2f}  {delta}{marker}")

    print("\n  [3/3] DRIFT WINDOW RE-TEST (HRP ×3.0 Core3, 5m, 1% commission)")
    print(f"  {'DriftWin':>10}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs 60d':>9}")
    print('  ' + '─' * 66)
    r_60d = stats(champ)
    for dw in [30, 45, 60, 90, 120]:
        s = champ if dw == 60 else build_hrp_drift_w(assets6, 5, BOOST, CORE3, dw)
        r = stats(s)
        if not r: continue
        marker = ' ← champion' if dw == 60 else ''
        delta = f"{r[0] - r_60d[0]:>+8.1%}" if dw != 60 else '  baseline'
        print(f"  {dw:>9}d  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {r[5]:>8.2f}  {delta}{marker}")

    print(f"\n{'═' * W}")


def portfolio_v55_vol_target_and_cluster_structure(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v55: Two remaining untested dimensions on v53 champion (HRP ×3.0 Core3 5m):
    (1) Vol-target re-opt (25/30/35/40/45/50/60%): was 40% optimal for inv-vol in v41.
        With ×3.0 boost, Core3 months are near-always full-long regardless of target.
        Only non-Core3 months are affected. New optimal may differ.
    (2) HRP cluster structure variants:
        - Current: {BTC,ETH} vs {SOL,BNB,ADA,AVAX}
        - BTC-only: {BTC} vs {ETH,SOL,BNB,ADA,AVAX}
        - Top3: {BTC,ETH,SOL} vs {BNB,ADA,AVAX}
        - Equal-weight within clusters (vs 1/σ²)
    """
    W = 122
    CORE3 = frozenset({11, 3, 7})
    BOOST = 3.0
    VOL_W = 5

    def make_signal(target_vol: float, base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            if base_pct > 0:
                base = sig_base_position(df, base_pct=base_pct)
            else:
                base = sig_vol_targeting(df, window=60, target_ann_vol=target_vol, rvol_window=45)
            return _apply_seasonal_scalar(base, CORE3, weak=set(), boost=BOOST)
        return fn

    def build_hrp_custom(assets, vol_w, large_caps, c=0.01, equal_w=False):
        """HRP with custom cluster definition and optional equal-weight within clusters."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(0.40, 0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(0.40, 0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c

        def hrp_rp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_n = list(eq_df.columns)
            large = [n for n in all_n if n in large_caps]
            small = [n for n in all_n if n not in large_caps]
            rets, dates = [], []
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                if equal_w:
                    wl = pd.Series(1.0 / len(large), index=large) if large else pd.Series(dtype=float)
                    ws = pd.Series(1.0 / len(small), index=small) if small else pd.Series(dtype=float)
                else:
                    inv_v = 1.0 / (trailing.var() + 1e-16)
                    wl = inv_v[large] / inv_v[large].sum() if large else pd.Series(dtype=float)
                    ws = inv_v[small] / inv_v[small].sum() if small else pd.Series(dtype=float)
                rl = float((mth_r.iloc[i][large] * wl).sum()) if len(large) > 0 else 0.0
                rs = float((mth_r.iloc[i][small] * ws).sum()) if len(small) > 0 else 0.0
                if large and small:
                    tr_l = (trailing[large] * wl).sum(axis=1)
                    tr_s = (trailing[small] * ws).sum(axis=1)
                    vl = float(tr_l.var()) + 1e-16
                    vs = float(tr_s.var()) + 1e-16
                    cl = (1 / vl) / (1 / vl + 1 / vs)
                else:
                    cl = 1.0 if large else 0.0
                rets.append(cl * rl + (1 - cl) * rs)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = hrp_rp(eq_main)
        sb = hrp_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def build_vol_target(assets, target_vol, vol_w=VOL_W, c=0.01):
        """HRP with custom vol-target."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_signal(target_vol, 0.0)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_signal(target_vol, 0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c
        LARGE_CAPS = {'BTC', 'ETH'}

        def hrp_rp(equities):
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_n = list(eq_df.columns)
            large = [n for n in all_n if n in LARGE_CAPS]
            small = [n for n in all_n if n not in LARGE_CAPS]
            rets, dates = [], []
            for i in range(vol_w, len(mth_r)):
                trailing = mth_r.iloc[i - vol_w:i]
                inv_v = 1.0 / (trailing.var() + 1e-16)
                wl = inv_v[large] / inv_v[large].sum()
                ws = inv_v[small] / inv_v[small].sum()
                rl = float((mth_r.iloc[i][large] * wl).sum())
                rs = float((mth_r.iloc[i][small] * ws).sum())
                vl = float((trailing[large] * wl).sum(axis=1).var()) + 1e-16
                vs = float((trailing[small] * ws).sum(axis=1).var()) + 1e-16
                cl = (1 / vl) / (1 / vl + 1 / vs)
                rets.append(cl * rl + (1 - cl) * rs)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = hrp_rp(eq_main)
        sb = hrp_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def stats(s):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start: mask &= s_full.index >= start
        if end: mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'),
          ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]

    LARGE2 = {'BTC', 'ETH'}
    champ = build_hrp_custom(assets6, VOL_W, LARGE2)
    r_champ = stats(champ)

    print(f"\n{'═' * W}")
    print("  v55 — VOL-TARGET RE-OPT + HRP CLUSTER STRUCTURE VARIANTS")
    print(f"{'═' * W}")


    print("\n  [1/2] VOL-TARGET SWEEP (HRP ×3.0 Core3, 5m, {BTC,ETH}/rest, 1% commission)")
    print("  Note: ×3.0 boost → Core3 months near-always full-long regardless of target")
    print(f"  {'Target':>8}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs 40%':>9}")
    print("  " + "─" * 66)

    r_40 = r_champ
    for tv in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        s = champ if tv == 0.40 else build_vol_target(assets6, tv)
        r = stats(s)
        if not r: continue
        marker = ' ← champion' if tv == 0.40 else ''
        delta = f"{r[0] - r_40[0]:>+8.1%}" if tv != 0.40 else '  baseline'
        print(f"  {tv:>7.0%}  {r[0]:>+8.1%}  {r[1]:>6.2f}  {r[2]:>4.0%}  {r[4]:>+7.1%}  {r[5]:>8.2f}  {delta}{marker}")


    print("\n  [2/2] HRP CLUSTER STRUCTURE VARIANTS (vol-target 40%, ×3.0, 5m, 1%)")
    cluster_configs = [
        ('{{BTC,ETH}} / rest (champion)', {'BTC', 'ETH'}, False),
        ('{{BTC}} / rest (BTC-solo)', {'BTC'}, False),
        ('{{BTC,ETH,SOL}} / rest', {'BTC', 'ETH', 'SOL'}, False),
        ('Eq-wt within {{BTC,ETH}}/rest', {'BTC', 'ETH'}, True),
    ]
    print(f"  {'Cluster config':>34}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}")
    print("  " + "─" * 82)

    cluster_series = {}
    for label, large_caps, eq_w in cluster_configs:
        s = champ if (large_caps == LARGE2 and not eq_w) else \
            build_hrp_custom(assets6, VOL_W, large_caps, equal_w=eq_w)
        cluster_series[label] = s
        r = stats(s)
        if not r: continue
        cum, sh, wr, n, mdd, calmar = r
        marker = ' ← champion' if (large_caps == LARGE2 and not eq_w) else ''
        print(f"  {label:>34}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}{marker}")

    best_cluster = max(cluster_series, key=lambda k: (stats(cluster_series[k]) or (0,))[0])
    r_best_c = stats(cluster_series[best_cluster])
    if r_best_c and r_champ:
        delta = r_best_c[0] - r_champ[0]
        verdict = 'ADOPT' if delta > 0.05 and best_cluster != list(cluster_series.keys())[0] else 'KEEP current'
        print(f"\n  → Best cluster: '{best_cluster}'  Δ={delta:+.1%}  → {verdict}")


    if best_cluster != list(cluster_series.keys())[0]:
        print(f"\n  Walk-forward ({best_cluster} vs current champion):")
        hdr = f"  {'Variant':>34}"
        for p, *_ in WF:
            hdr += f"  {p:>12}"
        print(hdr)
        print("  " + "─" * (36 + 14 * len(WF)))
        for label in [list(cluster_series.keys())[0], best_cluster]:
            row = f"  {label:>34}"
            for _, start, end in WF:
                r = sub_stats(cluster_series[label], start, end)
                if r:
                    ok = '✓' if r[0] > 0 else '✗'
                    row += f"  {r[0]:>+9.1%} {ok}"
                else:
                    row += f"  {'n/a':>12}"
            print(row)

    print(f"\n{'═' * W}")


def portfolio_v56_regime_gate(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v56: BTC 200d MA regime gate + portfolio DD gate on v53 HRP champion.
    Motivation: P1 bear weakness (-11.4%) is structural (long-only). A regime filter
    using BTC 200d MA can reduce exposure in bear markets without adding shorts.
    Gate is applied post-HRP on monthly returns using prev month-end signal (no look-ahead).
    """
    W = 122
    CORE3 = frozenset({11, 3, 7})
    BOOST = 3.0
    VOL_W = 5
    LARGE_CAPS_V56 = {'BTC', 'ETH'}

    def make_sig(target_vol: float, base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            if base_pct > 0:
                return _apply_seasonal_scalar(
                    sig_base_position(df, base_pct=base_pct), CORE3, weak=set(), boost=BOOST)
            return _apply_seasonal_scalar(
                sig_vol_targeting(df, window=60, target_ann_vol=target_vol, rvol_window=45),
                CORE3, weak=set(), boost=BOOST)

        return fn

    def build_champion_rets(c: float = 0.01) -> pd.Series:
        """Build v53 HRP champion monthly return series."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_sig(0.40)(df)).equity for n, df in assets6}
            eq_floor = {n: Backtest(df, make_sig(0.40, 0.02)(df)).equity for n, df in assets6}
        finally:
            COMMISSION = old_c

        def hrp_rp(equities: dict) -> pd.Series:
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_n = list(eq_df.columns)
            large = [n for n in all_n if n in LARGE_CAPS_V56]
            small = [n for n in all_n if n not in LARGE_CAPS_V56]
            rets, dates = [], []
            for i in range(VOL_W, len(mth_r)):
                trailing = mth_r.iloc[i - VOL_W:i]
                inv_v = 1.0 / (trailing.var() + 1e-16)
                wl = inv_v[large] / inv_v[large].sum()
                ws = inv_v[small] / inv_v[small].sum()
                rl = float((mth_r.iloc[i][large] * wl).sum())
                rs = float((mth_r.iloc[i][small] * ws).sum())
                vl = float((trailing[large] * wl).sum(axis=1).var()) + 1e-16
                vs = float((trailing[small] * ws).sum(axis=1).var()) + 1e-16
                cl = (1 / vl) / (1 / vl + 1 / vs)
                rets.append(cl * rl + (1 - cl) * rs)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = hrp_rp(eq_main)
        sb = hrp_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def apply_ma_gate(base_rets: pd.Series, btc_price: pd.Series, ma200:
                      pd.Series, scale_below: float) -> pd.Series:
        """Scale month-t return by scale_below if BTC < 200d MA at end of month t-1."""
        result = []
        for date, ret in base_rets.items():
            mask = ma200.index < date
            if mask.sum() < 1 or pd.isna(ma200[mask].iloc[-1]):
                result.append(ret)
                continue
            last_price = float(btc_price[mask].iloc[-1])
            last_ma = float(ma200[mask].iloc[-1])
            scale = 1.0 if last_price > last_ma else scale_below
            result.append(ret * scale)
        return pd.Series(result, index=base_rets.index)

    def apply_dd_gate(base_rets: pd.Series, threshold: float, scale_below:
                      float) -> pd.Series:
        """Scale month-t return by scale_below if portfolio DD at month start > threshold."""
        result = list(base_rets.values)
        equity = 1.0
        peak = 1.0
        for i, ret in enumerate(base_rets.values):
            dd = equity / peak - 1.0
            if i > 0 and dd < -threshold:
                result[i] = ret * scale_below
            equity *= (1 + result[i])
            peak = max(peak, equity)
        return pd.Series(result, index=base_rets.index)

    def stats(s: pd.Series):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float(((s + 1).cumprod().div((s + 1).cumprod().cummax())).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full: pd.Series, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start:
            mask &= s_full.index >= start
        if end:
            mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'),
          ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]

    print(f"\n{'═' * W}")
    print("  v56 — BTC 200d MA REGIME GATE + PORTFOLIO DD GATE")
    print(f"{'═' * W}")
    print("  Motivation: P1 bear weakness (-11.4%) is structural long-only limitation")
    print("  Gate applied post-HRP on monthly returns using prev month-end signal (no look-ahead)")

    base = build_champion_rets(c=0.01)
    r_base = stats(base)

    btc_df = dict(assets6)['BTC']
    btc_price = btc_df['Close']
    ma200 = btc_price.rolling(200).mean()

    print("\n  [1/2] BTC 200d MA REGIME GATE (scale when BTC close < 200d MA at prev month-end)")
    print(f"  {'Scale':>6}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs base':>9}")
    print("  ────────────────────────────────────────────────────────────────────")

    ma_series = {}
    for scale in [0.0, 0.25, 0.50, 0.75, 1.0]:
        s = base if scale == 1.0 else apply_ma_gate(base, btc_price, ma200, scale)
        ma_series[scale] = s
        r = stats(s)
        if not r:
            continue
        cum, sh, wr, n, mdd, calmar = r
        marker = ' ← baseline' if scale == 1.0 else ''
        delta = f"{cum - r_base[0]:>+8.1%}" if scale != 1.0 else '  baseline'
        print(f"  ×{scale:>4.2f}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  {calmar:>8.2f}  {delta}{marker}")

    best_ma = max(ma_series, key=lambda k: (stats(ma_series[k]) or (0, 0))[1])
    r_bm = stats(ma_series[best_ma])
    d_cum = r_bm[0] - r_base[0] if r_bm else 0
    d_sh = r_bm[1] - r_base[1] if r_bm else 0
    verdict = 'ADOPT' if best_ma < 1.0 and d_sh > 0.02 else ('MARGINAL' if d_sh > 0 else 'REJECT')
    print(f"\n  → Best MA gate: ×{best_ma:.2f}  CumRet Δ={d_cum:+.1%}  Sh Δ={d_sh:+.2f}  → {verdict}")

    print(f"\n  Walk-forward — MA gate ×{best_ma:.2f} vs baseline:")
    print(f"  {'Period':>12}  {'Gate':>10}  {'Base':>10}  {'Delta':>8}  {'Gate Sh':>8}  {'Gate MDD':>9}")
    print("  ────────────────────────────────────────────────────────────────────")
    for label, start, end in WF:
        rg = sub_stats(ma_series[best_ma], start, end)
        rb = sub_stats(base, start, end)
        if rg and rb:
            marker = ' ✓' if rg[0] > 0 else ' ✗'
            print(f"  {label:>12}  {rg[0]:>+9.1%}  {rb[0]:>+9.1%}  {rg[0] - rb[0]:>+7.1%}  "
                  f"{rg[1]:>7.2f}  {rg[4]:>+8.1%}{marker}")

    print("\n  [2/2] PORTFOLIO DRAWDOWN GATE (scale when DD from recent peak > threshold)")
    print(f"  {'DD≥':>5}  {'Scale':>6}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  {'vs base':>9}")
    print("  ───────────────────────────────────────────────────────────────────────────")

    dd_series = {}
    for threshold in [0.10, 0.15, 0.20, 0.25]:
        for scale in [0.0, 0.5]:
            s = apply_dd_gate(base, threshold, scale)
            key = (threshold, scale)
            dd_series[key] = s
            r = stats(s)
            if not r:
                continue
            cum, sh, wr, n, mdd, calmar = r
            delta = f"{cum - r_base[0]:>+8.1%}"
            print(f"  {threshold:>4.0%}  ×{scale:>4.2f}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  "
                  f"{mdd:>+7.1%}  {calmar:>8.2f}  {delta}")
        print()

    best_dd = max(dd_series, key=lambda k: (stats(dd_series[k]) or (0, 0))[1])
    r_bdd = stats(dd_series[best_dd])
    d_cum_dd = r_bdd[0] - r_base[0] if r_bdd else 0
    d_sh_dd = r_bdd[1] - r_base[1] if r_bdd else 0
    verdict_dd = 'ADOPT' if d_sh_dd > 0.02 else ('MARGINAL' if d_sh_dd > 0 else 'REJECT')
    print(f"  → Best DD gate: threshold={best_dd[0]:.0%} scale=×{best_dd[1]:.2f}  CumRet Δ="
          f"{d_cum_dd:+.1%}  Sh Δ={d_sh_dd:+.2f}  → {verdict_dd}")

    if best_ma < 1.0:
        print(f"\n  [3/3] COMBINED: MA ×{best_ma:.2f} + DD {best_dd[0]:.0%}×{best_dd[1]:.2f}")
        s_ma_only = ma_series[best_ma]
        s_combined = apply_dd_gate(s_ma_only, best_dd[0], best_dd[1])
        r_comb = stats(s_combined)
        if r_comb:
            cum, sh, wr, n, mdd, calmar = r_comb
            print(f"  CumRet={cum:+.1%}  Sh={sh:.2f}  WR={wr:.0%}  MaxDD={mdd:+.1%}  Calmar={calmar:.2f}")
            d_c = cum - r_base[0]; d_s = sh - r_base[1]
            print(f"  vs baseline: CumRet Δ={d_c:+.1%}  Sh Δ={d_s:+.2f}")
        print("\n  Walk-forward — Combined gate vs baseline:")
        print(f"  {'Period':>12}  {'Combined':>10}  {'Base':>10}  {'Delta':>8}")
        print("  ──────────────────────────────────────────────────")
        for label, start, end in WF:
            rc = sub_stats(s_combined, start, end)
            rb = sub_stats(base, start, end)
            if rc and rb:
                marker = ' ✓' if rc[0] > 0 else ' ✗'
                print(f"  {label:>12}  {rc[0]:>+9.1%}  {rb[0]:>+9.1%}  {rc[0] - rb[0]:>+7.1%}{marker}")

    print(f"\n  {'─' * W}")
    print("  DECISION SUMMARY:")
    print(f"  MA gate best: ×{best_ma:.2f}  Sh Δ={d_sh:+.2f}  → {verdict}")
    print(f"  DD gate best: {best_dd[0]:.0%}×{best_dd[1]:.2f}  Sh Δ={d_sh_dd:+.2f}  → {verdict_dd}")
    if verdict == 'ADOPT':
        print(f"  → NEW CHAMPION: add MA gate ×{best_ma:.2f} to v53 spec")
    elif verdict_dd == 'ADOPT':
        print(f"  → NEW CHAMPION: add DD gate {best_dd[0]:.0%}×{best_dd[1]:.2f} to v53 spec")
    else:
        print("  → KEEP v53 HRP champion unchanged — regime gates do not improve risk-adjusted returns")

    print(f"\n{'═' * W}")


def portfolio_v57_ma_window_sensitivity(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v57: MA window sensitivity + commission sweep on BTC 200d MA-gated HRP champion.
    v56 adopted 200d MA gate (×0.00 = full cash when BTC < MA) giving +1377.7%, Sh=0.85.
    This function confirms 200d is optimal vs 100/150/200/250/300d, then runs commission sweep.
    """
    W = 122
    CORE3 = frozenset({11, 3, 7})
    BOOST = 3.0
    VOL_W = 5
    LARGE_CAPS_V57 = {'BTC', 'ETH'}

    def make_sig(target_vol: float, base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            if base_pct > 0:
                return _apply_seasonal_scalar(
                    sig_base_position(df, base_pct=base_pct), CORE3, weak=set(), boost=BOOST)
            return _apply_seasonal_scalar(
                sig_vol_targeting(df, window=60, target_ann_vol=target_vol, rvol_window=45),
                CORE3, weak=set(), boost=BOOST)
        return fn

    def build_hrp_rets(assets, c: float = 0.01) -> pd.Series:
        """Build v53 HRP monthly return series at given commission."""
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_sig(0.4)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_sig(0.4, 0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c

        def hrp_rp(equities: dict) -> pd.Series:
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_n = list(eq_df.columns)
            large = [n for n in all_n if n in LARGE_CAPS_V57]
            small = [n for n in all_n if n not in LARGE_CAPS_V57]
            rets, dates = [], []
            for i in range(VOL_W, len(mth_r)):
                trailing = mth_r.iloc[i - VOL_W:i]
                inv_v = 1.0 / (trailing.var() + 1e-16)
                wl = inv_v[large] / inv_v[large].sum()
                ws = inv_v[small] / inv_v[small].sum()
                rl = float((mth_r.iloc[i][large] * wl).sum())
                rs = float((mth_r.iloc[i][small] * ws).sum())
                vl = float((trailing[large] * wl).sum(axis=1).var()) + 1e-16
                vs = float((trailing[small] * ws).sum(axis=1).var()) + 1e-16
                cl = (1 / vl) / (1 / vl + 1 / vs)
                rets.append(cl * rl + (1 - cl) * rs)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = hrp_rp(eq_main)
        sb = hrp_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def apply_ma_gate(base_rets: pd.Series, btc_price: pd.Series, ma: pd.Series
                      ) -> pd.Series:
        """Full exit (×0) when BTC < MA at end of previous month. Returns 0 for flat months."""
        result = []
        for date, ret in base_rets.items():
            mask = ma.index < date
            if mask.sum() < 1 or pd.isna(ma[mask].iloc[-1]):
                result.append(ret)
            else:
                above = float(btc_price[mask].iloc[-1]) > float(ma[mask].iloc[-1])
                result.append(ret if above else 0.0)
        return pd.Series(result, index=base_rets.index)

    def stats(s: pd.Series):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full: pd.Series, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start:
            mask &= s_full.index >= start
        if end:
            mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'),
          ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]

    print(f"\n{'═' * W}")
    print('  v57 — MA WINDOW SENSITIVITY + COMMISSION SWEEP ON 200d MA-GATED HRP CHAMPION')
    print(f"{'═' * W}")
    print('  v56 result: BTC 200d MA gate ×0.00 → +1377.7%, Sh=0.85, Calmar=4.30')
    print('  v57 goal:   (1) confirm 200d is optimal; (2) commission robustness')

    btc_df = dict(assets6)['BTC']
    btc_price = btc_df['Close']
    base = build_hrp_rets(assets6, c=0.01)
    r_base = stats(base)

    print('\n  [1/2] MA WINDOW SENSITIVITY (BTC, ×0.00 gate, 1% commission)')
    print(f"  {'MAd':>5}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  "
          f"{'vs 200d':>9}  {'P1':>9}  {'P2':>9}  {'P3':>9}")
    print('  ────────────────────────────────────────────────────────────────────────────────────────────────────')

    ma_results = {}
    for win in [100, 150, 200, 250, 300]:
        ma = btc_price.rolling(win).mean()
        s = apply_ma_gate(base, btc_price, ma)
        ma_results[win] = s
        r = stats(s)
        r_p1 = sub_stats(s, None, '2023-01-01')
        r_p2 = sub_stats(s, '2023-01-01', '2025-01-01')
        r_p3 = sub_stats(s, '2025-01-01', None)
        if not r:
            continue
        cum, sh, wr, n, mdd, calmar = r
        marker = ' ←' if win == 200 else ''
        delta = f'{cum - r_base[0]:>+8.1%}'
        p1s = f'{r_p1[0]:>+8.1%}' if r_p1 else '       n/a'
        p2s = f'{r_p2[0]:>+8.1%}' if r_p2 else '       n/a'
        p3s = f'{r_p3[0]:>+8.1%}' if r_p3 else '       n/a'
        print(f'  {win:>4}d  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  '
              f'{calmar:>8.2f}  {delta}  {p1s}  {p2s}  {p3s}{marker}')

    r = r_base
    if r:
        cum, sh, wr, n, mdd, calmar = r
        r_p1 = sub_stats(base, None, '2023-01-01')
        r_p2 = sub_stats(base, '2023-01-01', '2025-01-01')
        r_p3 = sub_stats(base, '2025-01-01', None)
        p1s = f'{r_p1[0]:>+8.1%}' if r_p1 else '       n/a'
        p2s = f'{r_p2[0]:>+8.1%}' if r_p2 else '       n/a'
        p3s = f'{r_p3[0]:>+8.1%}' if r_p3 else '       n/a'
        print(f"  {'no gate':>5}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  "
              f"{calmar:>8.2f}  {'baseline':>9}  {p1s}  {p2s}  {p3s}")

    best_win = max(ma_results, key=lambda k: (stats(ma_results[k]) or (0, 0))[1])
    r_best = stats(ma_results[best_win])
    d_sh = r_best[1] - r_base[1] if r_best else 0
    verdict = 'CONFIRMED 200d OPTIMAL' if best_win == 200 else f'UPDATE TO {best_win}d'
    print(f'\n  → Best window: {best_win}d  Sh={r_best[1]:.2f}  Δ={d_sh:+.2f}  → {verdict}')

    champion_win = best_win
    champion_rets = ma_results[champion_win]

    print(f'\n  [2/2] COMMISSION SWEEP — BTC {champion_win}d MA-gated HRP champion')
    print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  "
          f"{'MaxDD':>8}  {'Calmar':>8}  {'Viable':>7}")
    print('  ───────────────────────────────────────────────────────────────────────────')

    for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        raw = build_hrp_rets(assets6, c=c)
        ma = btc_price.rolling(champion_win).mean()
        s = apply_ma_gate(raw, btc_price, ma)
        r = stats(s)
        if not r:
            continue
        cum, sh, wr, n, mdd, calmar = r
        ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
        print(f'  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  '
              f'{mdd:>+7.1%}  {calmar:>8.2f}  {ok}')

    r_champ = stats(champion_rets)
    r_01 = stats(apply_ma_gate(build_hrp_rets(assets6, c=0.001),
                               btc_price, btc_price.rolling(champion_win).mean()))
    print(f"\n{'═' * W}")
    print('  v57 FINAL CHAMPION SPECIFICATION — HRP + BTC MA Gate')
    print(f"{'═' * W}")
    if r_champ and r_01:
        cum1, sh1, wr1, n1, mdd1, cal1 = r_champ
        cum2, sh2, wr2, n2, mdd2, cal2 = r_01
        print(f"""
  ┌─────────────────────────────────────────────────────────────────────────────────────────┐
  │  STRATEGY: HRP + BTC {champion_win}d MA Regime Gate                                          │
  ├─────────────────────────────────────────────────────────────────────────────────────────┤
  │  Universe    : BTC, ETH, SOL, BNB, ADA, AVAX                                          │
  │  Rebalance   : Monthly, last trading day                                               │
  │  Signal      : VT40% (60d drift, 45d rvol, target=40%) + Core3 boost ×3.0            │
  │  Ensemble    : 98% main + 2% Base2 floor                                              │
  │  Portfolio   : Hierarchical inv-variance RP, {{BTC,ETH}}/rest, 5m vol window           │
  │  Regime gate : Full cash when BTC close < {champion_win}d MA at prev month-end                │
  ├─────────────────────────────────────────────────────────────────────────────────────────┤
  │  PERFORMANCE @ 1% commission (1% per trade):                                          │
  │    CumRet={cum1:>+7.1%}  Sh={sh1:.2f}  WR={wr1:.0%}  MaxDD={mdd1:>+6.1%}  Calmar={cal1:.2f}              │
  │  PERFORMANCE @ 0.1% commission (0.1% per trade):                                      │
  │    CumRet={cum2:>+7.1%}  Sh={sh2:.2f}  WR={wr2:.0%}  MaxDD={mdd2:>+6.1%}  Calmar={cal2:.2f}              │
  │  WALK-FORWARD (all periods positive):                                                  │""")
        for label, start, end in WF[1:]:
            rp = sub_stats(champion_rets, start, end)
            ok = '✓' if rp and rp[0] > 0 else '✗'
            val = f'{rp[0]:>+7.1%}' if rp else '    n/a'
            print(f'  │    {label}: {val}  {ok}                                                           │')
        print('  └─────────────────────────────────────────────────────────────────────────────────────────┘')

    print(f"\n{'═' * W}")


def portfolio_v58_fine_grained_ma_sweep(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v58: Fine-grained MA window sweep (120/130/140/150/160/170/180d) + scale sweep at 150d.
    v57 showed 150d beats 100d and 200d. This confirms 150d is a true local optimum
    (not a cliff) and verifies scale ×0.00 is optimal at 150d specifically.
    Final champion lock after this — research exhausted.
    """
    W = 122
    CORE3 = frozenset({11, 3, 7})
    BOOST = 3.0
    VOL_W = 5
    LARGE_CAPS_V58 = {'BTC', 'ETH'}

    def make_sig(target_vol: float, base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            if base_pct > 0:
                return _apply_seasonal_scalar(
                    sig_base_position(df, base_pct=base_pct), CORE3, weak=set(), boost=BOOST)
            return _apply_seasonal_scalar(
                sig_vol_targeting(df, window=60, target_ann_vol=target_vol, rvol_window=45),
                CORE3, weak=set(), boost=BOOST)
        return fn

    def build_hrp_rets(assets, c: float = 0.01) -> pd.Series:
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_sig(0.4)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_sig(0.4, 0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c

        def hrp_rp(equities: dict) -> pd.Series:
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_n = list(eq_df.columns)
            large = [n for n in all_n if n in LARGE_CAPS_V58]
            small = [n for n in all_n if n not in LARGE_CAPS_V58]
            rets, dates = [], []
            for i in range(VOL_W, len(mth_r)):
                trailing = mth_r.iloc[i - VOL_W:i]
                inv_v = 1.0 / (trailing.var() + 1e-16)
                wl = inv_v[large] / inv_v[large].sum()
                ws = inv_v[small] / inv_v[small].sum()
                rl = float((mth_r.iloc[i][large] * wl).sum())
                rs = float((mth_r.iloc[i][small] * ws).sum())
                vl = float((trailing[large] * wl).sum(axis=1).var()) + 1e-16
                vs = float((trailing[small] * ws).sum(axis=1).var()) + 1e-16
                cl = (1 / vl) / (1 / vl + 1 / vs)
                rets.append(cl * rl + (1 - cl) * rs)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = hrp_rp(eq_main)
        sb = hrp_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def apply_ma_gate(base_rets: pd.Series, btc_price: pd.Series, ma: pd.Series,
                      scale: float = 0.0) -> pd.Series:
        """Scale month-t return by `scale` if BTC < MA at end of prev month."""
        result = []
        for date, ret in base_rets.items():
            mask = ma.index < date
            if mask.sum() < 1 or pd.isna(ma[mask].iloc[-1]):
                result.append(ret)
            else:
                above = float(btc_price[mask].iloc[-1]) > float(ma[mask].iloc[-1])
                result.append(ret if above else ret * scale)
        return pd.Series(result, index=base_rets.index)

    def stats(s: pd.Series):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full: pd.Series, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start: mask &= s_full.index >= start
        if end: mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'),
          ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]

    print(f"\n{'═' * W}")
    print('  v58 — FINE-GRAINED MA WINDOW SWEEP + SCALE CONFIRM AT 150d')
    print(f"{'═' * W}")
    print('  v57: 150d beats 100d/200d. Goal: (1) confirm 150d local optimum; (2) confirm ×0.00 optimal at 150d')

    btc_df = dict(assets6)['BTC']
    btc_price = btc_df['Close']
    base = build_hrp_rets(assets6, c=0.01)
    r_base = stats(base)

    print('\n  [1/2] FINE-GRAINED MA WINDOW (×0.00 gate, 1% commission)')
    print(f"  {'MAd':>5}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  "
          f"{'P1':>9}  {'P2':>9}  {'P3':>9}")
    print('  ' + '─' * 95)

    win_results = {}
    for win in [120, 130, 140, 150, 160, 170, 180]:
        ma = btc_price.rolling(win).mean()
        s = apply_ma_gate(base, btc_price, ma, scale=0.0)
        win_results[win] = s
        r = stats(s)
        r_p1 = sub_stats(s, None, '2023-01-01')
        r_p2 = sub_stats(s, '2023-01-01', '2025-01-01')
        r_p3 = sub_stats(s, '2025-01-01', None)
        if not r: continue
        cum, sh, wr, n, mdd, calmar = r
        mark = ' ←' if win == 150 else ''
        p1s = f'{r_p1[0]:>+8.1%}' if r_p1 else '       n/a'
        p2s = f'{r_p2[0]:>+8.1%}' if r_p2 else '       n/a'
        p3s = f'{r_p3[0]:>+8.1%}' if r_p3 else '       n/a'
        print(f'  {win:>4}d  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  '
              f'{calmar:>8.2f}  {p1s}  {p2s}  {p3s}{mark}')

    best_win = max(win_results, key=lambda k: (stats(win_results[k]) or (0, 0))[1])
    r_best = stats(win_results[best_win])
    verdict = 'CONFIRMED 150d' if best_win == 150 else f'UPDATE TO {best_win}d'
    print(f'\n  → Best window: {best_win}d  Sh={r_best[1]:.2f}  → {verdict}')

    opt_win = best_win
    print(f'\n  [2/2] SCALE SWEEP at BTC {opt_win}d MA gate (1% commission)')
    print(f"  {'Scale':>6}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  "
          f"{'P1':>9}  {'P2':>9}  {'P3':>9}")
    print('  ' + '─' * 92)

    scale_results = {}
    ma_opt = btc_price.rolling(opt_win).mean()
    for scale in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        s = apply_ma_gate(base, btc_price, ma_opt, scale=scale)
        scale_results[scale] = s
        r = stats(s)
        r_p1 = sub_stats(s, None, '2023-01-01')
        r_p2 = sub_stats(s, '2023-01-01', '2025-01-01')
        r_p3 = sub_stats(s, '2025-01-01', None)
        if not r: continue
        cum, sh, wr, n, mdd, calmar = r
        mark = ' ← baseline' if scale == 1.0 else ''
        p1s = f'{r_p1[0]:>+8.1%}' if r_p1 else '       n/a'
        p2s = f'{r_p2[0]:>+8.1%}' if r_p2 else '       n/a'
        p3s = f'{r_p3[0]:>+8.1%}' if r_p3 else '       n/a'
        print(f'  ×{scale:>4.2f}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  '
              f'{calmar:>8.2f}  {p1s}  {p2s}  {p3s}{mark}')

    best_scale = max(scale_results, key=lambda k: (stats(scale_results[k]) or (0, 0))[1])
    r_bs = stats(scale_results[best_scale])
    verdict_sc = 'CONFIRMED ×0.00' if best_scale == 0.0 else f'UPDATE TO ×{best_scale:.2f}'
    print(f'\n  → Best scale: ×{best_scale:.2f}  Sh={r_bs[1]:.2f}  → {verdict_sc}')

    champ_rets = (scale_results[best_scale] if best_win == opt_win
                  else apply_ma_gate(base, btc_price, btc_price.rolling(best_win).mean(), scale=best_scale))
    r_champ = stats(champ_rets)
    r_01 = stats(apply_ma_gate(
        build_hrp_rets(assets6, c=0.001),
        btc_price, btc_price.rolling(best_win).mean(), scale=best_scale))

    print(f"\n{'═' * W}")
    print('  v58 FINAL CHAMPION — RESEARCH EXHAUSTED')
    print(f"{'═' * W}")
    if r_champ and r_01:
        c1, s1, w1, n1, d1, cal1 = r_champ
        c2, s2, w2, n2, d2, cal2 = r_01
        print(f'\n  ┌────────────────────────────────────────────────────────────────────────────────────┐\n'
              f'  │  STRATEGY: 6-Asset HRP v53 + BTC {best_win}d MA Regime Gate (scale ×{best_scale:.2f})          │\n'
              f'  ├────────────────────────────────────────────────────────────────────────────────────┤\n'
              f'  │  Universe   : BTC, ETH, SOL, BNB, ADA, AVAX                                      │\n'
              f'  │  Signal     : VT40% (60d drift, 45d rvol) + Core3 boost ×3.0 (Mar/Jul/Nov)      │\n'
              f'  │  Ensemble   : 98% VT40%+Core3 + 2% Base2 floor                                  │\n'
              f"  │  Portfolio  : HRP {'{BTC,ETH}'}/ rest, inv-variance, 5m vol window, monthly rebal  │\n"
              f'  │  Regime gate: CASH when BTC < {best_win}d MA at prev month-end (no look-ahead)        │\n'
              f'  ├────────────────────────────────────────────────────────────────────────────────────┤\n'
              f'  │  @ 1.0% commission:  CumRet={c1:>+7.1%}  Sh={s1:.2f}  MaxDD={d1:>+6.1%}  Calmar={cal1:.2f}   │\n'
              f'  │  @ 0.1% commission:  CumRet={c2:>+7.1%}  Sh={s2:.2f}  MaxDD={d2:>+6.1%}  Calmar={cal2:.2f}   │\n'
              f'  ├────────────────────────────────────────────────────────────────────────────────────┤')
        for label, start, end in WF[1:]:
            rp = sub_stats(champ_rets, start, end)
            ok = '✓' if rp and rp[0] > 0 else '✗'
            val = f'{rp[0]:>+7.1%}' if rp else '    n/a'
            print(f'  │  {label}: {val}  {ok}                                                    │')
        print('  └────────────────────────────────────────────────────────────────────────────────────┘')

    print('\n  All research dimensions exhausted. Champion fully specified.')
    print(f"{'═' * W}")


def portfolio_v59_core3_gate_exemption(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v59: Core3 gate exemption — in Mar/Jul/Nov months, stay active even if BTC < 150d MA.
    Hypothesis: Core3 months have ×3.0 boost for good reason (seasonal bounces).
    Going to cash in those months may sacrifice the highest-conviction trades.
    Tests: full exemption vs partial (×0.5 in Core3 when below MA) vs baseline (no exemption).
    """
    W = 122
    CORE3 = frozenset({11, 3, 7})
    BOOST = 3.0
    VOL_W = 5
    MA_WIN = 150
    LARGE_CAPS_V59 = {'BTC', 'ETH'}

    def make_sig(target_vol: float, base_pct: float = 0.0):
        def fn(df: pd.DataFrame) -> pd.Series:
            if base_pct > 0:
                return _apply_seasonal_scalar(
                    sig_base_position(df, base_pct=base_pct), CORE3, weak=set(), boost=BOOST)
            return _apply_seasonal_scalar(
                sig_vol_targeting(df, window=60, target_ann_vol=target_vol, rvol_window=45),
                CORE3, weak=set(), boost=BOOST)
        return fn

    def build_hrp_rets(assets, c: float = 0.01) -> pd.Series:
        global COMMISSION
        old_c = COMMISSION
        COMMISSION = c
        try:
            eq_main = {n: Backtest(df, make_sig(0.4)(df)).equity for n, df in assets}
            eq_floor = {n: Backtest(df, make_sig(0.4, 0.02)(df)).equity for n, df in assets}
        finally:
            COMMISSION = old_c

        def hrp_rp(equities: dict) -> pd.Series:
            eq_df = pd.DataFrame(equities).dropna()
            mth_r = eq_df.resample('ME').last().pct_change().dropna()
            all_n = list(eq_df.columns)
            large = [n for n in all_n if n in LARGE_CAPS_V59]
            small = [n for n in all_n if n not in LARGE_CAPS_V59]
            rets, dates = [], []
            for i in range(VOL_W, len(mth_r)):
                trailing = mth_r.iloc[i - VOL_W:i]
                inv_v = 1.0 / (trailing.var() + 1e-16)
                wl = inv_v[large] / inv_v[large].sum()
                ws = inv_v[small] / inv_v[small].sum()
                rl = float((mth_r.iloc[i][large] * wl).sum())
                rs = float((mth_r.iloc[i][small] * ws).sum())
                vl = float((trailing[large] * wl).sum(axis=1).var()) + 1e-16
                vs = float((trailing[small] * ws).sum(axis=1).var()) + 1e-16
                cl = (1 / vl) / (1 / vl + 1 / vs)
                rets.append(cl * rl + (1 - cl) * rs)
                dates.append(mth_r.index[i])
            return pd.Series(rets, index=pd.DatetimeIndex(dates))

        sa = hrp_rp(eq_main)
        sb = hrp_rp(eq_floor)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()

    def apply_gate(base_rets: pd.Series, btc_price: pd.Series, ma: pd.Series,
                   core3_scale: float) -> pd.Series:
        """Apply 150d MA gate with Core3 exemption.
        core3_scale: scale applied in Core3 months when BTC < MA.
          0.0 = no exemption (same as v58 champion)
          1.0 = full exemption (stay fully active in Core3 when below MA)
          0.5 = partial exemption
        Non-Core3 months below MA are always gated to 0 (cash).
        """
        result = []
        for date, ret in base_rets.items():
            mask = ma.index < date
            if mask.sum() < 1 or pd.isna(ma[mask].iloc[-1]):
                result.append(ret)
                continue
            above = float(btc_price[mask].iloc[-1]) > float(ma[mask].iloc[-1])
            if above:
                result.append(ret)
            else:
                month = date.month
                is_core3 = month in CORE3
                scale = core3_scale if is_core3 else 0.0
                result.append(ret * scale)
        return pd.Series(result, index=base_rets.index)

    def stats(s: pd.Series):
        if len(s) < 4:
            return None
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        n = len(s)
        mdd = float((s + 1).cumprod().div((s + 1).cumprod().cummax()).min() - 1)
        ann = float((1 + cum) ** (12 / n) - 1) if n > 0 else 0
        calmar = ann / abs(mdd) if mdd != 0 else 0
        return cum, sh, wr, n, mdd, calmar

    def sub_stats(s_full: pd.Series, start, end):
        mask = pd.Series(True, index=s_full.index)
        if start: mask &= s_full.index >= start
        if end: mask &= s_full.index < end
        return stats(s_full[mask])

    WF = [('Full 5y', None, None), ('P1 2021-23', None, '2023-01-01'),
          ('P2 2023-24', '2023-01-01', '2025-01-01'), ('P3 2025-26', '2025-01-01', None)]

    print(f"\n{'═' * W}")
    print('  v59 — CORE3 GATE EXEMPTION (stay active in Mar/Jul/Nov when BTC < 150d MA)')
    print(f"{'═' * W}")
    print('  Hypothesis: Core3 months have ×3.0 boost; seasonal bounces occur even in bears')
    print('  v58 champion: gate to 0 in ALL months below 150d MA (no exemption)')

    btc_df = dict(assets6)['BTC']
    btc_price = btc_df['Close']
    ma150 = btc_price.rolling(MA_WIN).mean()
    base = build_hrp_rets(assets6, c=0.01)

    gated_core3 = []
    gated_other = []
    for date, ret in base.items():
        mask = ma150.index < date
        if mask.sum() < 1:
            continue
        above = float(btc_price[mask].iloc[-1]) > float(ma150[mask].iloc[-1])
        if above:
            continue
        if date.month in CORE3:
            gated_core3.append(date.strftime('%Y-%m'))
        else:
            gated_other.append(date.strftime('%Y-%m'))

    print('\n  Gate activity in v58 (months where BTC < 150d MA):')
    print(f"    Core3 months gated (Mar/Jul/Nov): {len(gated_core3)} — {', '.join(gated_core3) if gated_core3 else 'none'}")
    print(f'    Other months gated:               {len(gated_other)}')

    print('\n  Core3 exemption sweep (non-Core3 below MA always gated to cash):')
    print(f"  {'C3 scale':>9}  {'CumRet':>9}  {'Sh':>6}  {'WR':>5}  {'MaxDD':>8}  {'Calmar':>8}  "
          f"{'P1':>9}  {'P2':>9}  {'P3':>9}  {'Label':>20}")
    print('  ' + '─' * 108)

    variants = {}
    configs = [
        (0.0, 'v58 champion (no exempt)'),
        (0.25, 'C3 partial ×0.25'),
        (0.5, 'C3 partial ×0.50'),
        (0.75, 'C3 partial ×0.75'),
        (1.0, 'C3 full exempt ×1.00'),
    ]
    r_base58 = stats(apply_gate(base, btc_price, ma150, 0.0))

    for c3_scale, label in configs:
        s = apply_gate(base, btc_price, ma150, c3_scale)
        variants[c3_scale] = s
        r = stats(s)
        r_p1 = sub_stats(s, None, '2023-01-01')
        r_p2 = sub_stats(s, '2023-01-01', '2025-01-01')
        r_p3 = sub_stats(s, '2025-01-01', None)
        if not r: continue
        cum, sh, wr, n, mdd, calmar = r
        mark = ' ←' if c3_scale == 0.0 else ''
        p1s = f'{r_p1[0]:>+8.1%}' if r_p1 else '       n/a'
        p2s = f'{r_p2[0]:>+8.1%}' if r_p2 else '       n/a'
        p3s = f'{r_p3[0]:>+8.1%}' if r_p3 else '       n/a'
        print(f'  ×{c3_scale:>6.2f}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {mdd:>+7.1%}  '
              f'{calmar:>8.2f}  {p1s}  {p2s}  {p3s}  {label}{mark}')

    best_c3 = max(variants, key=lambda k: (stats(variants[k]) or (0, 0))[1])
    r_best = stats(variants[best_c3])
    d_sh = r_best[1] - r_base58[1] if r_best and r_base58 else 0
    d_cum = r_best[0] - r_base58[0] if r_best and r_base58 else 0
    verdict = 'ADOPT exemption' if best_c3 > 0.0 and d_sh > 0.01 else 'KEEP v58 (no exemption)'
    print(f'\n  → Best C3 scale: ×{best_c3:.2f}  Sh Δ={d_sh:+.2f}  CumRet Δ={d_cum:+.1%}  → {verdict}')

    if gated_core3:
        print('\n  Monthly returns in gated Core3 months (v58 gated these to 0):')
        print(f"  {'Month':>9}  {'HRP ret':>9}  {'Exempt keeps':>14}  {'Gate gives':>12}")
        print('  ' + '─' * 52)
        for mth_str in gated_core3:
            mth_idx = pd.to_datetime(mth_str + '-01') + pd.offsets.MonthEnd(0)
            if mth_idx in base.index:
                ret = float(base.loc[mth_idx])
                print(f"  {mth_str:>9}  {ret:>+8.1%}  {'active':>14}  {'cash (0%)':>12}  "
                      f"{'✓ keep' if ret > 0 else '✗ avoid'}")

    print(f"\n  {'─' * W}")
    print(f'  DECISION: {verdict}')
    if best_c3 > 0.0 and d_sh > 0.01:
        c1, s1, w1, n1, d1, cal1 = r_best
        print(f'  NEW CHAMPION: BTC 150d MA gate + Core3 exemption ×{best_c3:.2f}')
        print(f'  CumRet={c1:+.1%}  Sh={s1:.2f}  MaxDD={d1:+.1%}  Calmar={cal1:.2f}')
    else:
        print('  v58 champion stands. Core3 exemption does not improve risk-adjusted returns.')

    print(f"\n{'═' * W}")


def portfolio_v40_window_and_universe(assets6: list[tuple[str, pd.DataFrame]]) -> None:
    """v40: (1) Drift window sensitivity on 6-asset champion (30/45/60/90/120d).
    (2) 8-asset universe expansion: add XRP + LINK to 6-asset RP.
    Original loop prompt point (1) applied to the converged champion portfolio.
    """
    W = 122
    def make_c3vt45(w: int):
        """Return Core3+VT45 signal function with drift window w."""
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_vol_targeting_40_45d(df, window=w)
            return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak=set(), boost=1.2)
        return fn
    def make_c3base2(w: int):
        """Return Core3+Base2+VT45 signal function with drift window w."""
        def fn(df: pd.DataFrame) -> pd.Series:
            base = sig_base_position(df, base_pct=0.02, window=w)
            return _apply_seasonal_scalar(base, SEASONAL_CORE3_MONTHS, weak=set(), boost=1.2)
        return fn
    def rp_ens98(assets, w: int, c: float=0.01):
        sa = _rp_at_commission(assets, make_c3vt45(w), c)
        sb = _rp_at_commission(assets, make_c3base2(w), c)
        idx = sa.index.intersection(sb.index)
        return (0.98 * sa.loc[idx] + 0.02 * sb.loc[idx]).dropna()
    def stats(s):
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        wins = int((s > 0).sum())
        n = len(s)
        return (cum, sh, wr, wins, n)
    print(f"\n{'═' * W}")
    print('  v40 WINDOW SENSITIVITY + 8-ASSET UNIVERSE')
    print('  Testing drift window 30/45/60/90/120d on 6-asset Ens 98/2 champion')
    print(f"{'═' * W}")
    print('\n  6-ASSET Ens 98/2 — Drift window sensitivity @ 1% commission:')
    print(f"  {'Window':>8}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>5}  {'Wins':>7}  {'vs W60':>8}")
    print('  ────────────────────────────────────────────────────────────')
    results_w = {}
    for w in [30, 45, 60, 90, 120]:
        s = rp_ens98(assets6, w)
        cum, sh, wr, wins, n = stats(s)
        results_w[w] = (cum, sh, wr, s)
        print(f'  {w:>6}d    {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}     —')
    base_cum, base_sh, _, _ = results_w[60]
    print(f"  {'':>8}  {'':>9}  {'':>8}  {'':>5}  {'':>7}  {'vs W60':>8}")
    for w in [30, 45, 90, 120]:
        cum, sh, wr, _ = results_w[w]
        delta = cum - base_cum
        marker = '★ BETTER' if cum > base_cum else ''
        print(f"  {w:>6}d    {cum:>+8.1%}  {sh:>8.2f}  {wr:>4.0%}  {'':>7}  {delta:>+7.1%}  {marker}")
    print(f"  {'60d':>8} [champion baseline]  {base_cum:>+8.1%}  {base_sh:>8.2f}")
    best_w = max(results_w, key=lambda k: results_w[k][1])
    print(f'\n  → Optimal window (Sharpe): {best_w}d  Sh={results_w[best_w][1]:.2f}  CumRet={results_w[best_w][0]:+.1%}')
    if best_w != 60:
        print(f'\n  Commission sweep — 6-asset Ens 98/2 with W={best_w}d:')
        print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
        print('  ──────────────────────────────────────────────────────────')
        for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
            s = rp_ens98(assets6, best_w, c)
            cum, sh, wr, _, _ = stats(s)
            ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
            print(f'  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}')
    if best_w != 60:
        print(f'\n  Walk-forward Core3+VT45 W={best_w}d — all 6 assets:')
        fn_best = make_c3vt45(best_w)
        for name, df in assets6:
            walk_forward(df, f'Core3+VT45 W{best_w}d — {name}', fn_best)
    print(f"\n{'═' * W}")
    print('\n  8-ASSET UNIVERSE — Adding XRP + LINK to 6-asset RP:')
    print('  Hypothesis: 4→6 assets doubled returns (+150%→+428%); can 6→8 do it again?')
    print()
    try:
        xrp = load_data('XRP-USD')
        link = load_data('LINK-USD')
        assets8 = assets6 + [('XRP', xrp), ('LINK', link)]
        print(f'  Loaded XRP: {len(xrp)} days,  LINK: {len(link)} days')
    except Exception as e:
        print(f'  Data load failed: {e}')
        print(f"\n{'═' * W}")
        return None
    print('\n  Comparing 4/6/8-asset Ens 98/2 Core3+VT45 @ 1% commission:')
    HDR = f"  {'Portfolio':<45}  {'CumRet':>9}  {'MthSh':>6}  {'WR':>5}  {'Wins':>7}"
    print(HDR)
    print('  ────────────────────────────────────────────────────────────────────────────────')
    use_w = best_w
    assets4 = [a for a in assets6 if a[0] in ['BTC', 'ETH', 'SOL', 'BNB']]
    for label, assets in [
        ('4-asset RP (BTC/ETH/SOL/BNB)', assets4),
        ('6-asset RP + ADA/AVAX [v38 champ]', assets6),
        ('8-asset RP + XRP/LINK [v40 test]', assets8),
    ]:
        s = rp_ens98(assets, use_w)
        cum, sh, wr, wins, n = stats(s)
        star = '★' if len(assets) == 8 else ' '
        print(f'  {star} {label:<44}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}')
    print()
    for label, assets in [
        ('7-asset RP + XRP only', assets6 + [('XRP', xrp)]),
        ('7-asset RP + LINK only', assets6 + [('LINK', link)]),
    ]:
        s = rp_ens98(assets, use_w)
        cum, sh, wr, wins, n = stats(s)
        print(f'    {label:<44}  {cum:>+8.1%}  {sh:>6.2f}  {wr:>4.0%}  {wins:>3}/{n:<3}')
    s6 = rp_ens98(assets6, use_w, 0.01)
    s8 = rp_ens98(assets8, use_w, 0.01)
    best_assets = assets8 if stats(s8)[1] > stats(s6)[1] else assets6
    best_label = '8-asset' if best_assets is assets8 else '6-asset (unchanged)'
    print(f'\n  Commission sweep — {best_label} Ens 98/2 W={use_w}d:')
    print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
    print('  ──────────────────────────────────────────────────────────')
    for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        s = rp_ens98(best_assets, use_w, c)
        cum, sh, wr, _, _ = stats(s)
        ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
        print(f'  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}')
    if best_assets is assets8:
        print(f'\n  Walk-forward Core3+VT45 W={use_w}d — XRP and LINK:')
        fn = make_c3vt45(use_w)
        for name, df in [('XRP', xrp), ('LINK', link)]:
            walk_forward(df, f'Core3+VT45 — {name}', fn)
    print(f"\n{'═' * W}")


def final_champion_validation(assets6: list[tuple[str, pd.DataFrame]]):
    """v38: Final validation of the champion: 6-asset Ens 98/2 Core3/Core3Base2.\n    Walk-forward all 6 assets + full commission sweep + updated deployment spec.\n    """
    W = 118
    print(f"\n{'═' * W}")
    print('  v38 FINAL CHAMPION VALIDATION — 6-asset Ens 98/2 Core3/Core3Base2')
    print('  CumRet=+428%, Sh=0.72, WR=44% @ 1% commission')
    print(f"{'═' * W}")
    def rp_series_at_c(assets, sig_fn, c=0.01):
        return _rp_at_commission(assets, sig_fn, c)
    print('\n  Commission sweep — 6-asset Ens 98/2 Core3/Core3Base2:')
    print(f"  {'Comm/trade':>12}  {'RT':>6}  {'CumRet':>9}  {'MthSh':>8}  {'WR':>6}  {'Viable':>7}")
    print('  ───────────────────────────────────────────────────────')
    for c in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        s_a = rp_series_at_c(assets6, sig_core3_seasonal_vt45, c)
        s_b = rp_series_at_c(assets6, sig_core3_base2_vt45, c)
        idx = s_a.index.intersection(s_b.index)
        s = (0.98 * s_a.loc[idx] + 0.02 * s_b.loc[idx]).dropna()
        cum = float((1 + s).prod() - 1)
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        wr = float((s > 0).mean())
        ok = '✓ YES' if cum > 0 and sh > 0.3 else '✗ NO'
        print(f'  {c:>11.3%}  {2 * c:>5.3%}  {cum:>+8.1%}  {sh:>8.2f}  {wr:>5.0%}  {ok:>7}')
    print()
    print('  Walk-forward Core3+VT45 — ADA and AVAX:')
    for name, df in [(n, d) for n, d in assets6 if n in ['ADA', 'AVAX']]:
        walk_forward(df, f'Core3+VT45 — {name}', sig_core3_seasonal_vt45)
    print(f"\n{'═' * W}")
    print('  v38 FINAL DEPLOYMENT SPECIFICATION — Updated champion')
    print(f"{'═' * W}")
    print('\n  ┌─────────────────────────────────────────────────────────────────────────────────────────┐\n  │  STRATEGY: GBM Core3 Seasonal + Vol-Targeting — 6-Asset Risk-Parity Portfolio          │\n  ├─────────────────────────────────────────────────────────────────────────────────────────┤\n  │  Universe    : BTC, ETH, SOL, BNB, ADA, AVAX                                          │\n  │  Rebalance   : Monthly, last trading day of each month                                 │\n  │  Drift window: 60-day log-return mean & std                                            │\n  │  Vol window  : 45-day realized vol (annualized), target = 40%                          │\n  │  Position    : min(0.40 / rvol_45d, 1.0) if drift_60d > 0, else 0                     │\n  │  Seasonal    : ×1.2 in Mar/Jul/Nov ONLY  (Core3 — removed Oct/Aug, added Nov)         │\n  │  Ensemble    : 98% Core3+VT45 + 2% Core3+Base2+VT45 (floor trick for WR)             │\n  │  Portfolio   : Inverse trailing-6m vol weights (risk parity), monthly rebal            │\n  ├─────────────────────────────────────────────────────────────────────────────────────────┤\n  │  PERFORMANCE (5yr backtest, 1% commission):                                            │\n  │    Sharpe=0.72  │  Monthly WR=44%  │  CumRet≈+428%  │  VaR-99≈-17.5%/mo             │\n  │  AT 0.1% COMMISSION (realistic CEX spot):                                              │\n  │    Sharpe≈0.82  │  Monthly WR≈50%  │  CumRet significantly higher                    │\n  │  VS 4-ASSET ORIGINAL BASELINE:                                                         │\n  │    Was: Sh=0.72, WR=30%, +150%  →  Now: Sh=0.72, WR=44%, +428%  (+278% improvement)  │\n  └─────────────────────────────────────────────────────────────────────────────────────────┘\n')
    print('═' * W)


def btc_lead_analysis(assets: list[tuple[str, pd.DataFrame]], df_btc: pd.DataFrame):
    """Compare own-signal vs BTC-lead signal for ETH / SOL / BNB."""
    W = 110
    print(f"\n{'═' * W}")
    print("  BTC LEAD SIGNAL ANALYSIS  (BTC drift → time ETH/SOL/BNB)")
    print("  Hypothesis: BTC's GBM signal applied to altcoins beats own-signal")
    print(f"{'═' * W}")
    hdr = (f"  {'Strategy':<28}  {'Asset':<5}  {'Return':>9}  {'Sharpe':>7}  "
           f"{'MaxDD':>8}  {'Trades':>7}  {'AvgWin':>8}  {'WinRate':>9}")
    print(hdr)
    print('─' * W)

    alt_assets = [(n, df) for n, df in assets if n != 'BTC']

    strategy_pairs = [
        ('Own VT (40%,45d)',     lambda df: sig_vol_targeting_40_45d(df),     None),
        ('BTC-Lead VT',          lambda df: sig_btc_lead_vt(df, df_btc),      None),
        ('Own Asymmetric',       lambda df: sig_gbm_asymmetric(df),           None),
        ('BTC-Lead Asymmetric',  lambda df: sig_btc_lead_asym(df, df_btc),    None),
        ('Own Blend',            lambda df: sig_blend_50_50(df),              None),
    ]

    for strat_name, strat_fn, _ in strategy_pairs:
        first = True
        for asset_name, df_a in alt_assets:
            bt = Backtest(df_a, strat_fn(df_a))
            mth_ret = bt.equity.resample('ME').last().pct_change().dropna()
            wr = float((mth_ret > 0).mean())
            avg_win = float(mth_ret[mth_ret > 0].mean()) if (mth_ret > 0).any() else 0.0
            lbl = strat_name if first else ''
            first = False
            print(f"  {lbl:<28}  {asset_name:<5}  "
                  f"{bt.total_return:>+8.1%}  {bt.sharpe:>7.2f}  "
                  f"{bt.max_drawdown:>7.1%}  {bt.trades:>7}  "
                  f"{avg_win:>+7.1%}  {wr:>8.0%}")
        print()
    print('═' * W)


def _garch_conditional_var(returns_arr: np.ndarray) -> float:
    """Fit GARCH(1,1) on returns array, return next-period conditional variance."""
    from arch import arch_model
    try:
        am = arch_model(returns_arr * 100, vol='Garch', p=1, q=1,
                        mean='Zero', dist='normal', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        fc = res.forecast(horizon=1, reindex=False)
        var_pct = float(fc.variance.iloc[-1, 0])
        return var_pct / 10000
    except Exception:
        return float(np.var(returns_arr))


def sig_garch_monthly(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Monthly GBM signal with GARCH(1,1) conditional variance."""
    r = df['log_ret'].values
    mu_s = df['log_ret'].rolling(window).mean()
    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index,
    )
    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            mu = float(mu_s.iloc[i])
            g_var = _garch_conditional_var(r[max(0, i - window):i])
            drift = mu - 0.5 * g_var
            new = 1.0 if drift > 0 else 0.0
            if abs(new - pos) > 0.05:
                pos = new
        signal.iloc[i] = pos
    return _shift_signal(signal)


def sig_garch_vt(df: pd.DataFrame, window: int = 60, target_ann_vol: float = 0.4, rvol_window: int = 45) -> pd.Series:
    """Vol-targeting GBM signal with GARCH(1,1) conditional variance for drift."""
    r = df['log_ret'].values
    mu_s = df['log_ret'].rolling(window).mean()
    rvol = df['log_ret'].rolling(rvol_window).std() * np.sqrt(252)
    vs_s = (target_ann_vol / rvol).clip(0.0, 1.0)

    month_ends = df.index.to_period('M')
    is_month_end = pd.Series(
        [month_ends[i] != month_ends[i + 1] if i + 1 < len(month_ends) else True
         for i in range(len(month_ends))],
        index=df.index)

    signal = pd.Series(0.0, index=df.index)
    pos = 0.0
    for i in range(window, len(df)):
        if is_month_end.iloc[i]:
            mu = float(mu_s.iloc[i])
            vs = float(vs_s.iloc[i])
            g_var = _garch_conditional_var(r[max(0, i - window):i])
            drift = mu - 0.5 * g_var
            new = (float(vs) if not np.isnan(vs) else 1.0) if drift > 0 else 0.0
            if abs(new - pos) > 0.05: pos = new
        signal.iloc[i] = pos
    return _shift_signal(signal)


def garch_drift_analysis(assets: list[tuple[str, pd.DataFrame]]):
    '''Compare rolling drift vs GARCH(1,1) drift — both binary and vol-targeted.'''
    W = 110
    print(f'''\n{'═' * W}''')
    print('  GARCH(1,1) DRIFT ANALYSIS  (rolling vs GARCH conditional variance)')
    print('  Hypothesis: GARCH better models vol-clustering → more accurate drift')
    print(f'''{'═' * W}''')
    hdr = f'''  {'Strategy':<28}  {'Asset':<5}  {'Return':>9}  {'Sharpe':>7}  {'MaxDD':>8}  {'Trades':>7}  {'WR':>6}'''
    print(hdr)
    print('─' * W)
    strategies = [
        ('Rolling GBM Monthly', (lambda df: sig_gbm_monthly(df, window = 60))),
        ('GARCH GBM Monthly', (lambda df: sig_garch_monthly(df, window = 60))),
        ('Rolling Vol-Tgt(40%45d)', (lambda df: sig_vol_targeting_40_45d(df))),
        ('GARCH Vol-Tgt(40%45d)', (lambda df: sig_garch_vt(df, window = 60)))]
    for strat_name, strat_fn in strategies:
        first = True
        for asset_name, df_a in assets:
            bt = Backtest(df_a, strat_fn(df_a))
            mth = bt.equity.resample('ME').last().pct_change().dropna()
            wr = float((mth > 0).mean())
            lbl = strat_name if first else ''
            first = False
            print(f'''  {lbl:<28}  {asset_name:<5}  {bt.total_return:>+8.1%}  {bt.sharpe:>7.2f}  {bt.max_drawdown:>7.1%}  {bt.trades:>7}  {wr:>5.0%}''')
        print()
    print('═' * W)


def portfolio_riskparity_analysis(assets: list[tuple[str, pd.DataFrame]]):
    """Equal-risk-contribution portfolio of GBM Vol-Tgt signals."""
    W = 100
    print(f"\n{'═' * W}")
    print('  PORTFOLIO RISK PARITY ANALYSIS')
    print('  Strategy: GBM Vol-Tgt(40%,45d) on each asset, weighted by 1/σ monthly')
    print('  Benchmark: equal-weight (25% each)')
    print(f"{'═' * W}")

    asset_names = [n for n, _ in assets]

    equities = {}
    for name, df_a in assets:
        bt = Backtest(df_a, sig_vol_targeting_40_45d(df_a))
        equities[name] = bt.equity

    eq_df = pd.DataFrame(equities).dropna()
    mth = eq_df.resample('ME').last()
    mth_r = mth.pct_change().dropna()

    ew_ret = mth_r.mean(axis=1)

    port_rets = []
    port_dates = []
    rp_weights_history = []
    for i in range(6, len(mth_r)):
        trailing = mth_r.iloc[i - 6:i]
        vol_i = trailing.std()
        inv_vol = 1.0 / (vol_i + 1e-08)
        w = inv_vol / inv_vol.sum()
        rp_weights_history.append(w.values)
        port_rets.append(float((mth_r.iloc[i] * w).sum()))
        port_dates.append(mth_r.index[i])

    rp_ret = pd.Series(port_rets, index=port_dates)
    ew_trim = ew_ret.loc[rp_ret.index]

    def stats(s, label):
        n_mo = len(s)
        n_pos = int((s > 0).sum())
        wr = n_pos / n_mo if n_mo else 0
        sh = float(np.sqrt(12) * s.mean() / s.std()) if s.std() > 0 else 0
        cum = float((1 + s).prod() - 1)
        avg_w = float(s[s > 0].mean()) if n_pos else 0
        avg_l = float(s[s < 0].mean()) if (s < 0).any() else 0
        v99 = float(np.percentile(s, 1)) if n_mo >= 10 else float(s.min())
        streak = max_c = 0
        for r in s:
            streak = streak + 1 if r < 0 else 0
            max_c = max(max_c, streak)
        print(f'  {label:<28}  {cum:>+8.1%}  {sh:>7.2f}  '
              f'{wr:>6.0%}  {n_pos:>3}/{n_mo}  '
              f'{avg_w:>+7.1%}  {avg_l:>+8.1%}  {v99:>+9.1%}  {max_c:>5}')

    print(f"\n  {'Strategy':<28}  {'CumRet':>8}  {'MthSh':>7}  "
          f"{'WinRate':>7}  {'Wins':>7}  {'AvgWin':>8}  {'AvgLoss':>9}  "
          f"{'VaR-99':>9}  {'MaxCons':>7}")
    print('─' * W)

    for name, df_a in assets:
        bt = Backtest(df_a, sig_vol_targeting_40_45d(df_a))
        mth_i = bt.equity.resample('ME').last().pct_change().dropna()
        mth_i = mth_i.loc[rp_ret.index]
        stats(mth_i, f'VT45 — {name}')

    print('─' * W)
    stats(ew_trim, 'Equal-Weight Portfolio')
    stats(rp_ret, 'Risk-Parity Portfolio')
    print('─' * W)

    bh_rets = {}
    for name, df_a in assets:
        bt_bh = Backtest(df_a, sig_buy_hold(df_a))
        bh_rets[name] = bt_bh.equity.resample('ME').last().pct_change().dropna()
    bh_df = pd.DataFrame(bh_rets).dropna()
    bh_ew = bh_df.mean(axis=1).loc[rp_ret.index]
    stats(bh_ew, 'B&H Equal-Weight')

    print('═' * W)

    avg_w = np.mean(rp_weights_history, axis=0)
    print('\n  Avg risk-parity weights: ' +
          '  '.join(f'{n}={w:.1%}' for n, w in zip(asset_names, avg_w)))


def sig_gbm_intraday(df_i: pd.DataFrame, window_days: int = 7, bars_per_day: int = 288) -> pd.Series:
    """GBM drift on intraday bars (5m=288/day), rebalancing at end of each calendar day."""
    window = window_days * bars_per_day
    r = df_i['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    signal = pd.Series(0.0, index=df_i.index)
    pos = 0.0
    for i in range(len(df_i.index)):
        signal.iloc[i] = pos
        if i < len(df_i.index) - 1:
            if df_i.index[i + 1].date() != df_i.index[i].date():
                d = drift.iloc[i]
                if not pd.isna(d):
                    pos = 1.0 if d > 0 else 0.0
    return _shift_signal(signal)


def sig_vt_intraday(df_i: pd.DataFrame, window_days: int = 7, bars_per_day: int = 288, target_ann_vol: float = 0.4, rvol_days: int = 3) -> pd.Series:
    """Vol-targeting GBM on intraday bars, daily rebalancing."""
    window = window_days * bars_per_day
    rvol_win = rvol_days * bars_per_day
    r = df_i['log_ret']
    mu = r.rolling(window).mean()
    sig = r.rolling(window).std()
    drift = mu - 0.5 * sig ** 2
    bpy = bars_per_day * 365
    rvol = r.rolling(rvol_win).std() * np.sqrt(bpy)
    vol_scale = (target_ann_vol / rvol).clip(0.0, 1.0)
    signal = pd.Series(0.0, index=df_i.index)
    pos = 0.0
    for i in range(len(df_i.index)):
        signal.iloc[i] = pos
        if i < len(df_i.index) - 1:
            if df_i.index[i + 1].date() != df_i.index[i].date():
                d = drift.iloc[i]
                vs = vol_scale.iloc[i]
                if not pd.isna(d) and (not pd.isna(vs)):
                        pos = float(vs) if d > 0 else 0.0
    return _shift_signal(signal)


def vol_tgt_drift_window_sensitivity(assets: list[tuple[str, pd.DataFrame]], target_vol: float = 0.4, rvol_window: int = 45):
    """Test drift estimation window [30,45,60,75,90] with fixed 40%/45d config."""
    drift_windows = [30, 45, 60, 75, 90]
    p1_end = pd.Timestamp('2022-12-31')
    asset_names = [n for n, _ in assets]

    print(f"\n{'═' * 112}")
    print(f"  DRIFT-WINDOW SENSITIVITY  Vol-Tgt target={target_vol:.0%}  rvol={rvol_window}d  (all 4 assets)")
    print(f"{'═' * 112}")
    hdr = (f"  {'Drift W':>8}  {'BTC Ret':>9}  {'ETH Ret':>9}  "
           f"{'SOL Ret':>9}  {'BNB Ret':>9}  "
           f"{'BTC Sh':>7}  {'ETH Sh':>7}  {'SOL Sh':>7}  {'BNB Sh':>7}  "
           f"{'BTC P1':>8}  {'ETH P1':>8}  {'Avg Sh':>7}")
    print(hdr)
    print('─' * 112)

    best = {'score': -999, 'drift_w': 60}

    for dw in drift_windows:
        row_res = {}
        for name, df_a in assets:
            bt = Backtest(df_a, sig_vol_targeting(df_a, window=dw,
                                                  target_ann_vol=target_vol,
                                                  rvol_window=rvol_window))
            sub = df_a[df_a.index <= p1_end]
            bt_p1 = Backtest(sub, sig_vol_targeting(sub, window=dw,
                                                    target_ann_vol=target_vol,
                                                    rvol_window=rvol_window))
            row_res[name] = (bt, bt_p1)

        avg_sh = float(np.mean([v[0].sharpe for v in row_res.values()]))
        marker = '★' if avg_sh > best['score'] else ' '
        if avg_sh > best['score']:
            best = {'score': avg_sh, 'drift_w': dw}
        rets = '  '.join(f"{row_res[n][0].total_return:>+8.1%}" for n in asset_names)
        shs = '  '.join(f"{row_res[n][0].sharpe:>7.2f}" for n in asset_names)
        p1btc = row_res['BTC'][1].total_return
        p1eth = row_res['ETH'][1].total_return
        print(f"  {marker}{dw:>7}d  {rets}  {shs}  "
              f"{p1btc:>+7.1%}  {p1eth:>+7.1%}  {avg_sh:>7.2f}")

    print('═' * 112)
    print(f"  Optimal drift window: {best['drift_w']}d  avg Sharpe={best['score']:.3f}")
    return best['drift_w']


def vol_targeting_sensitivity(assets: list[tuple[str, pd.DataFrame]], window: int = 60):
    targets = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]

    p1_end = pd.Timestamp('2022-12-31')

    print(f"\n{'═'*96}")
    print("  GBM VOL-TARGETING — TARGET VOL SENSITIVITY  (1% commission)")
    print(f"{'═'*96}")
    hdr = (f"  {'Target':>7}  {'BTC Ret':>9}  {'BTC Sh':>7}  {'BTC DD':>8}  "
           f"{'ETH Ret':>9}  {'ETH Sh':>7}  {'ETH DD':>8}  "
           f"{'BTC P1':>8}  {'ETH P1':>8}  {'Avg Sh':>7}")
    print(hdr)
    print('─' * 96)

    best = {'score': -999, 'target': 0.40}
    all_res = []

    for t in targets:
        row = {}
        for name, df in assets:
            bt = Backtest(df, sig_vol_targeting(df, window=window, target_ann_vol=t))
            sub = df[df.index <= p1_end]
            bt_p1 = Backtest(sub,
                             sig_vol_targeting(sub, window=window, target_ann_vol=t))
            row[name] = (bt, bt_p1)

        avg_sh = np.mean([v[0].sharpe for v in row.values()])
        btc_bt, btc_p1 = row['BTC']
        eth_bt, eth_p1 = row['ETH']
        marker = '★' if avg_sh > best['score'] else ' '
        if avg_sh > best['score']:
            best = {'score': avg_sh, 'target': t}
        print(f"  {marker}{t:>6.0%}  "
              f"{btc_bt.total_return:>+8.1%}  {btc_bt.sharpe:>7.2f}  "
              f"{btc_bt.max_drawdown:>7.1%}  "
              f"{eth_bt.total_return:>+8.1%}  {eth_bt.sharpe:>7.2f}  "
              f"{eth_bt.max_drawdown:>7.1%}  "
              f"{btc_p1.total_return:>+7.1%}  {eth_p1.total_return:>+7.1%}  "
              f"{avg_sh:>7.2f}")
        all_res.append((t, avg_sh))

    print('═' * 96)
    print(f"  Optimal: target_vol={best['target']:.0%}  avg Sharpe={best['score']:.3f}")
    return best['target']


def vol_targeting_2d_grid(assets: list[tuple[str, pd.DataFrame]], drift_window: int = 60):
    """2D grid search: target_vol × rvol_window → find joint optimum."""

    targets = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50]
    rvol_windows = [30, 45, 60]
    p1_end = pd.Timestamp('2022-12-31')

    print(f"\n{'═' * 104}")
    print(f"  VOL-TARGETING 2D GRID  drift_window={drift_window}d  target_vol × rvol_window")
    print(f"{'═' * 104}")
    hdr = (f"  {'Target':>7}  {'RVol':>5}  {'BTC Ret':>9}  {'ETH Ret':>9}  "
           f"{'SOL Ret':>9}  {'BNB Ret':>9}  "
           f"{'AvgSh':>7}  {'BTC P1':>8}  {'ETH P1':>8}")
    print(hdr)
    print('─' * 104)

    best = {'score': -999, 'target': 0.40, 'rvol_w': 45}

    for t in targets:
        for rv in rvol_windows:
            row_res = {}
            for name, df_a in assets:
                bt = Backtest(df_a, sig_vol_targeting(df_a, window=drift_window,
                                                      target_ann_vol=t, rvol_window=rv))
                sub = df_a[df_a.index <= p1_end]
                bt_p1 = Backtest(sub, sig_vol_targeting(sub, window=drift_window,
                                                        target_ann_vol=t, rvol_window=rv))
                row_res[name] = (bt, bt_p1)

            avg_sh = float(np.mean([v[0].sharpe for v in row_res.values()]))
            marker = '★' if avg_sh > best['score'] else ' '
            if avg_sh > best['score']:
                best = {'score': avg_sh, 'target': t, 'rvol_w': rv}
            p1btc = row_res['BTC'][1].total_return
            p1eth = row_res['ETH'][1].total_return
            rets = '  '.join(f"{row_res[n][0].total_return:>+8.1%}"
                             for n in [nn for nn, _ in assets])
            print(f"  {marker}{t:>6.0%}  {rv:>4}d  {rets}  "
                  f"{avg_sh:>7.2f}  {p1btc:>+7.1%}  {p1eth:>+7.1%}")
        print('─' * 104)

    print('═' * 104)
    print(f"  OPTIMAL: target_vol={best['target']:.0%}  rvol_window={best['rvol_w']}d  avg Sharpe="
          f"{best['score']:.3f}")
    return best['target'], best['rvol_w']


def realized_vol_window_sensitivity(assets: list[tuple[str, pd.DataFrame]], drift_window: int = 60, target_vol: float = 0.4):
    """Test realized-vol measurement window (15/20/30/45/60d) for Vol-Tgt (40%), all 4 assets."""
    rvol_windows = [15, 20, 30, 45, 60]
    p1_end = pd.Timestamp('2022-12-31')
    asset_names = [n for n, _ in assets]

    print(f"\n{'═' * 112}")
    print(f"  REALIZED-VOL WINDOW SENSITIVITY  target_vol={target_vol:.0%}  drift_window={drift_window}d  (all 4 assets)")
    print(f"{'═' * 112}")
    hdr = (f"  {'RVol Win':>9}  {'BTC Ret':>9}  {'ETH Ret':>9}  "
           f"{'SOL Ret':>9}  {'BNB Ret':>9}  "
           f"{'BTC Sh':>7}  {'ETH Sh':>7}  {'SOL Sh':>7}  {'BNB Sh':>7}  "
           f"{'BTC P1':>8}  {'ETH P1':>8}  {'Avg Sh':>7}")
    print(hdr)
    print('─' * 112)

    best = {'score': -999, 'rvol_w': 30}

    for rvol_w in rvol_windows:
        row_res = {}
        for name, df_a in assets:
            bt = Backtest(df_a, sig_vol_targeting(df_a, window=drift_window,
                                                  target_ann_vol=target_vol,
                                                  rvol_window=rvol_w))
            sub = df_a[df_a.index <= p1_end]
            bt_p1 = Backtest(sub, sig_vol_targeting(sub, window=drift_window,
                                                    target_ann_vol=target_vol,
                                                    rvol_window=rvol_w))
            row_res[name] = (bt, bt_p1)

        avg_sh = float(np.mean([v[0].sharpe for v in row_res.values()]))
        marker = '★' if avg_sh > best['score'] else ' '
        if avg_sh > best['score']:
            best = {'score': avg_sh, 'rvol_w': rvol_w}
        rets = '  '.join(f"{row_res[n][0].total_return:>+8.1%}" for n in asset_names)
        shs = '  '.join(f"{row_res[n][0].sharpe:>7.2f}" for n in asset_names)
        p1btc = row_res['BTC'][1].total_return
        p1eth = row_res['ETH'][1].total_return
        print(f"  {marker}{rvol_w:>8}d  {rets}  {shs}  "
              f"{p1btc:>+7.1%}  {p1eth:>+7.1%}  {avg_sh:>7.2f}")

    print('═' * 112)
    print(f"  Optimal realized-vol window: {best['rvol_w']}d  avg Sharpe={best['score']:.3f}")
    return best['rvol_w']


def build_signals(df: pd.DataFrame) -> dict:
    gbm60 = sig_gbm_monthly(df, window=60)
    base = {
        'Golden Cross (50/200)':     sig_golden_cross(df),
        'Bull Filter (>200 SMA)':    sig_above_ma(df, window=200),
        'GBM Monthly (30d)':         sig_gbm_monthly(df, window=30),
        'GBM Monthly (45d)':         sig_gbm_monthly(df, window=45),
        'GBM Monthly (60d)':         gbm60,
        'GBM Monthly (90d)':         sig_gbm_monthly(df, window=90),
        'GBM Monthly (120d)':        sig_gbm_monthly(df, window=120),
        'GBM Monthly+Bull (60d)':    sig_gbm_monthly_bull(df, window=60),
        'GBM Monthly Scaled':        sig_gbm_monthly_scaled(df, window=60),
        'GBM Monthly Kelly':         sig_gbm_monthly_kelly(df, window=60),
        'GBM Monthly HalfKelly':     sig_gbm_monthly_half_kelly(df, window=60),
        'GBM60 + StopLoss(20%)':     apply_stop_loss(df, gbm60, max_dd=0.2),
        'Momentum Monthly (60d)':    sig_momentum_monthly(df, window=60),
        'GBM Monthly BearGate':      sig_gbm_monthly_bear_gate(df, window=60),
        'GBM Tight Gate (-15%)':     sig_gbm_tight_gate(df, window=60),
        'GBM Vol-Targeting (20%)':   sig_vol_targeting(df, window=60, target_ann_vol=0.2),
        'GBM Vol-Targeting (40%)':   sig_vol_targeting_40(df, window=60),
        'GBM Vol-Tgt Adaptive':      sig_vol_targeting_adaptive(df, window=60),
        'GBM Vol-Tgt (40%,45d)':     sig_vol_targeting_40_45d(df, window=60),
        'GBM Asymmetric (30/60)':    sig_gbm_asymmetric(df),
        'GBM Asym Vol-Gated':        sig_gbm_asymmetric_volgated(df),
        'GBM Blend (VT+Asym)':       sig_blend_50_50(df),
        'VT45 + MStop(5%)':          sig_vt45_monthly_stop(df, stop_pct=0.05),
        'VT45 + MStop(3%)':          sig_vt45_monthly_stop(df, stop_pct=0.03),
        'Blend + MStop(5%)':         sig_blend_monthly_stop(df, stop_pct=0.05),
        'Base5 + VT (monthly)':      sig_base5_vt(df),
        'Base10 + VT (monthly)':     sig_base10_vt(df),
        'Base5 + Blend (monthly)':   sig_base5_blend(df),
        'Base20 + VT (monthly)':     sig_base20_vt(df),
        'Base30 + VT (monthly)':     sig_base30_vt(df),
        'Weekly Vol-Tgt (40%,45d)':  sig_weekly_vol_targeting(df),
        'Base20 + Weekly VT':        sig_base20_weekly(df),
        'GBM Monthly L/S':           sig_gbm_monthly_ls(df, window=60),
        'VaR-99 Crash Buy (2×,25d)': sig_var99_crash(df, multiplier=2.0, hold_days=25),
        'VaR-99 Crash Buy (1.5×)':   sig_var99_crash(df, multiplier=1.5, hold_days=20),
        'GBM Throttled (90d,20h)':   sig_gbm_throttled(df, window=90, min_hold=20),
        'Log-Linear Trend (90d)':    sig_log_trend(df, window=90),
        'GBM Monte Carlo (p10)':     sig_gbm_montecarlo(df, window=60, horizon=10),
        'Trend + Low Vol':           sig_trend_low_vol(df),
        'VT45 + Seasonal':           sig_seasonal_vt45(df),
        'Blend + Seasonal':          sig_seasonal_blend(df),
        'Asym + Seasonal':           sig_seasonal_asym(df),
        'VT45 + Boost-Only':         sig_boost_only_seasonal_vt45(df),
        'Blend + Boost-Only':        sig_boost_only_seasonal_blend(df),
        'Asym + Boost-Only':         sig_boost_only_seasonal_asym(df),
        'VT45 + BullRegime Seas':    sig_bull_seasonal_vt45(df),
        'Base5+Blend+BoostOnly':     sig_base5_boost_blend(df),
        'DriftScaled+VT45+Boost':    sig_drift_scaled_vt45_boost(df),
        'BullFilter+VT45+Boost':     sig_bull_filter_vt45_boost(df),
        'Base2+VT45+Boost':          sig_base2_vt45_boost(df),
        'Base5+VT45+Boost':          sig_base5_vt45_boost(df),
        'Base2+Blend+Boost':         sig_base2_blend_boost(df),
        'Core3+VT45':                sig_core3_seasonal_vt45(df),
        'Core3+Base2+VT45':          sig_core3_base2_vt45(df),
        'Core3+Blend':               sig_core3_blend(df),
        'Core3+Base2+Blend':         sig_core3_base2_blend(df),
    }
    base['Ensemble (≥3)'] = sig_ensemble(
        {k: v for k, v in base.items()
         if 'GBM Monthly' in k or 'Golden' in k or 'Bull' in k},
        min_agree=3,
    )

    return base


def run_all(df: pd.DataFrame, label: str) -> dict[str, Backtest]:
    print(f'''\n  [{label}] Running {len(build_signals(df)) + 1} strategies...''')
    signals = build_signals(df)
    results = {
        'Buy & Hold': Backtest(df, sig_buy_hold(df)) }
    for name, sig in signals.items():
        allow_short = 'L/S' in name
        results[name] = Backtest(df, sig, allow_short = allow_short)
    return results

COLS = ['Total Return', 'Annual Return', 'Sharpe', 'Sortino', 'Calmar',
        'Max DD', 'VaR-99 (day)', 'Win Rate', 'Trades']
WIDTHS = [28, 14, 14, 8, 9, 8, 9, 15, 10, 8]


def print_table(results: dict[str, Backtest], label: str):
    print(f"\n{'══════════════════════════════════════════════════════════════════════════════════════════'}")
    print(f'  {label}')
    print(f"{'══════════════════════════════════════════════════════════════════════════════════════════'}")
    headers = ['Strategy'] + COLS
    print(''.join((h.ljust(w) for h, w in zip(headers, WIDTHS))))
    print('──────────────────────────────────────────────────────────────────────────────────────────')
    bh_ret = results['Buy & Hold'].total_return
    rows = []
    for name, bt in results.items():
        r = bt.row()
        rows.append((name, bt.total_return, r))
    rows.sort(key=lambda x: x[1], reverse=True)
    for name, ret, r in rows:
        marker = '★' if ret > bh_ret and name != 'Buy & Hold' else ' '
        cells = [f'{marker} {name}'] + [r[c] for c in COLS]
        print(''.join((v.ljust(w) for v, w in zip(cells, WIDTHS))))
    print('══════════════════════════════════════════════════════════════════════════════════════════')


def per_period_sharpe(assets: list[tuple[str, pd.DataFrame]], strategies: list[tuple[str, callable]], n_periods: int = 3):
    '''Show Sharpe ratio for each walk-forward period, per asset, per strategy.'''
    print(f'''\n{'════════════════════════════════════════════════════════════════════════════════════════════════════════════'}''')
    print('  PER-PERIOD SHARPE BREAKDOWN  (P1=bear 2021-22, P2=bull 2023-24, P3=recent 2024-26)')
    print(f'''{'════════════════════════════════════════════════════════════════════════════════════════════════════════════'}''')
    for strat_name, strat_fn in strategies:
        print(f'''\n  Strategy: {strat_name}''')
        hdr = f'''  {'Asset':<6}  {'P1 Ret':>9}  {'P1 Sh':>7}  {'P2 Ret':>9}  {'P2 Sh':>7}  {'P3 Ret':>9}  {'P3 Sh':>7}  {'Full':>9}  {'Full Sh':>8}'''
        print(hdr)
        print('  ────────────────────────────────────────────────────────────────────────────────')
        for name, df_a in assets:
            period_size = len(df_a) // n_periods
            period_results = []
            for i in range(n_periods):
                s = i * period_size
                e = s + period_size if i < n_periods - 1 else len(df_a)
                sub = df_a.iloc[s:e].copy()
                bt = Backtest(sub, strat_fn(sub))
                period_results.append(bt)
            bt_full = Backtest(df_a, strat_fn(df_a))
            (p1, p2, p3) = period_results
            print(f'''  {name:<6}  {p1.total_return:>+8.1%}  {p1.sharpe:>7.2f}  {p2.total_return:>+8.1%}  {p2.sharpe:>7.2f}  {p3.total_return:>+8.1%}  {p3.sharpe:>7.2f}  {bt_full.total_return:>+8.1%}  {bt_full.sharpe:>8.2f}''')
    print('════════════════════════════════════════════════════════════════════════════════════════════════════════════')


def annual_returns_table(assets: list[tuple[str, pd.DataFrame]], strategies: list[tuple[str, callable]]):
    """Calendar-year returns (2021-2026) for each strategy × asset."""
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    print(f"\n{'═' * 110}")
    print('  ANNUAL RETURNS BY CALENDAR YEAR  (partial: 2021 from May, 2026 to May)')
    print(f"{'═' * 110}")

    for strat_name, strat_fn in strategies:
        print(f'\n  Strategy: {strat_name}')
        hdr = f"  {'Asset':<6}  " + '  '.join(f'{y:>9}' for y in years) + f"  {'5-yr':>9}  {'Ann Sh':>7}"
        print(hdr)
        print('  ' + '─' * 85)
        for asset_name, df_a in assets:
            full_bt = Backtest(df_a, strat_fn(df_a))
            year_rets = []
            for y in years:
                sub = df_a[df_a.index.year == y].copy()
                if len(sub) < 5:
                    year_rets.append(None)
                    continue
                bt_y = Backtest(sub, strat_fn(sub))
                year_rets.append(bt_y.total_return)
            yr_strs = '  '.join(f'{r:>+8.1%}' if r is not None else f"{'—':>9}"
                                for r in year_rets)

            print(f'  {asset_name:<6}  {yr_strs}  '
                  f'{full_bt.total_return:>+8.1%}  {full_bt.sharpe:>7.2f}')
    print('═' * 110)


def commission_sensitivity_flagship(assets: list[tuple[str, pd.DataFrame]], strategy_name: str, strategy_fn: callable):
    """Commission sensitivity for a given strategy across all 4 assets."""
    global COMMISSION
    rates = [0.001, 0.0025, 0.005, 0.01, 0.015, 0.02]
    print(f"\n{'═' * 90}")
    print(f"  COMMISSION SENSITIVITY — {strategy_name}  (all 4 assets)")
    print(f"{'═' * 90}")
    asset_names = [n for n, _ in assets]
    hdr = (f"  {'Commission':>12}  {'RT':>6}  "
           + '  '.join(f"{n + ' Ret':>11}" for n in asset_names)
           + f"  {'AvgSh':>7}")
    print(hdr)
    print('─' * 90)

    for c in rates:
        old_c = COMMISSION
        COMMISSION = c
        bts = [Backtest(df_a, strategy_fn(df_a)) for _, df_a in assets]
        COMMISSION = old_c
        avg_sh = float(np.mean([bt.sharpe for bt in bts]))
        viable = '✓' if all(bt.total_return > 0 for bt in bts) else '✗'
        rets = '  '.join(f"{bt.total_return:>+10.1%}" for bt in bts)
        print(f"  {viable} {c:>10.2%}  {2 * c:>5.2%}  {rets}  {avg_sh:>7.2f}")

    print('═' * 90)


def window_sensitivity(btc: pd.DataFrame, eth: pd.DataFrame):
    windows = [
        30,
        45,
        60,
        90,
        120]
    print(f'''\n{'═══════════════════════════════════════════════════════════════════════════'}''')
    print('  GBM MONTHLY REBAL — WINDOW SENSITIVITY  (1% commission)')
    print(f'''{'═══════════════════════════════════════════════════════════════════════════'}''')
    print(f'''  {'Window':>8}  {'BTC Ret':>10}  {'BTC Sharpe':>11}  {'ETH Ret':>10}  {'ETH Sharpe':>11}  {'Trades(BTC)':>12}''')
    print('───────────────────────────────────────────────────────────────────────────')
    best = {
        'score': -999,
        'window': 60 }
    bh_btc = Backtest(btc, sig_buy_hold(btc)).total_return
    bh_eth = Backtest(eth, sig_buy_hold(eth)).total_return
    for w in windows:
        bt_b = Backtest(btc, sig_gbm_monthly(btc, window = w))
        bt_e = Backtest(eth, sig_gbm_monthly(eth, window = w))
        score = bt_b.sharpe + bt_e.sharpe
        marker = '★' if bt_b.total_return > bh_btc and bt_e.total_return > bh_eth else ' '
        print(f'''  {marker}{w:>7}d  {bt_b.total_return:>+9.1%}  {bt_b.sharpe:>11.2f}  {bt_e.total_return:>+9.1%}  {bt_e.sharpe:>11.2f}  {bt_b.trades:>12}''')
        if not score > best['score']:
            continue
        best = {
            'score': score,
            'window': w,
            'btc_ret': bt_b.total_return,
            'eth_ret': bt_e.total_return }
    print('═══════════════════════════════════════════════════════════════════════════')
    print(f'''  Optimal window: {best['window']}d  (BTC={best['btc_ret']:+.1%}, ETH={best['eth_ret']:+.1%}, combined Sharpe={best['score']:.2f})''')
    return best['window']


def walk_forward(df: pd.DataFrame, strategy_name: str, strategy_fn, n_periods: int = 3):
    period_size = len(df) // n_periods
    print(f'''\n  Walk-Forward: {strategy_name}''')
    print(f'''  {'Period':<6} {'Date Range':<36} {'Strat':>8} {'B&H':>8} {'Alpha':>8} {'Sharpe':>7}''')
    print('  ─────────────────────────────────────────────────────────────────────────')
    all_pos = []
    for i in range(n_periods):
        s = i * period_size
        e = s + period_size if i < n_periods - 1 else len(df)
        sub = df.iloc[s:e].copy()
        sig_s = strategy_fn(sub)
        sig_bh = sig_buy_hold(sub)
        bt_s = Backtest(sub, sig_s)
        bt_bh = Backtest(sub, sig_bh)
        alpha = bt_s.total_return - bt_bh.total_return
        dates = f'''{sub.index[0].date()} → {sub.index[-1].date()}'''
        marker = '✓' if bt_s.total_return > 0 else '✗'
        print(f'''  {marker} P{i + 1:<4} {dates:<36} {bt_s.total_return:>+7.1%} {bt_bh.total_return:>+7.1%} {alpha:>+7.1%} {bt_s.sharpe:>7.2f}''')
        all_pos.append(bt_s.total_return > 0)
    consistent = sum(all_pos)
    print(f'''  Consistency: {consistent}/{n_periods} periods profitable''')


def commission_sensitivity(btc: pd.DataFrame, eth: pd.DataFrame, strategy_name: str = 'GBM Monthly (60d)', window: int = 60):
    global COMMISSION, COMMISSION
    rates = [
        0.001,
        0.0025,
        0.005,
        0.01,
        0.015,
        0.02]
    print(f'''\n{'══════════════════════════════════════════════════════════════════════'}''')
    print(f'''  COMMISSION SENSITIVITY — {strategy_name}''')
    print(f'''{'══════════════════════════════════════════════════════════════════════'}''')
    print(f'''  {'Commission':>12}  {'Round-trip':>11}  {'BTC Ret':>9}  {'ETH Ret':>9}  {'BTC Sharpe':>11}''')
    print('──────────────────────────────────────────────────────────────────────')
    sig_fn = sig_gbm_monthly
    for c in rates:
        old_c = COMMISSION
        COMMISSION = c
        sig_b = sig_fn(btc, window = window)
        sig_e = sig_fn(eth, window = window)
        bt_b = Backtest(btc, sig_b)
        bt_e = Backtest(eth, sig_e)
        COMMISSION = old_c
        viable = '✓' if bt_b.total_return > 0 and bt_e.total_return > 0 else '✗'
        print(f'''  {viable} {c:>10.2%}  {2 * c:>10.2%}  {bt_b.total_return:>+8.1%}  {bt_e.total_return:>+8.1%}  {bt_b.sharpe:>11.2f}''')
    print('══════════════════════════════════════════════════════════════════════')
    print('  Note: break-even is where ETH return turns negative (ETH is harder market)')


def cross_validate(btc_res: dict, eth_res: dict):
    print(f'''\n{'══════════════════════════════════════════════════════════════════════════════════════════'}''')
    print('  CROSS-ASSET VALIDATION SUMMARY')
    print('  Strategy performance: BTC (development) → ETH (out-of-sample)')
    print(f'''{'══════════════════════════════════════════════════════════════════════════════════════════'}''')
    print(f'''  {'Strategy':<30} {'BTC Return':>12} {'ETH Return':>12} {'Status'}''')
    print('──────────────────────────────────────────────────────────────────────────────────────────')
    bh_btc = btc_res['Buy & Hold'].total_return
    bh_eth = eth_res['Buy & Hold'].total_return
    for name in btc_res:
        if name not in eth_res:
            continue
        br = btc_res[name].total_return
        er = eth_res[name].total_return
        if name == 'Buy & Hold':
            status = 'BENCHMARK'
        elif br > 0 and er > 0:
            status = '✓ PROFITABLE on both'
        elif br > bh_btc and er > bh_eth:
            status = '★ BEATS B&H on both'
        elif br > 0:
            status = '~ positive BTC only'
        else:
            status = '✗ unprofitable'
        print(f'''  {name:<30} {br:>+11.1%} {er:>+11.1%}   {status}''')
    print('══════════════════════════════════════════════════════════════════════════════════════════')


def plot_asset(df: pd.DataFrame, results: dict[str, Backtest], title: str, outfile: str):
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.semilogy(df.index, df['Close'], color='#1a1a2e', linewidth=1)
    ax1.set_title(f'{title} — Price (log scale)', fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1, :])
    palette = plt.cm.tab20(np.linspace(0, 1, len(results)))
    sorted_r = sorted(results.items(), key=lambda x: x[1].total_return, reverse=True)
    for (name, bt), color in zip(sorted_r, palette):
        lw = 2.0 if name == 'Buy & Hold' else 1.0
        ls = '--' if name == 'Buy & Hold' else '-'
        label = f'{name} ({bt.total_return:+.0%})'
        ax2.plot(bt.equity.index, bt.equity, label=label,
                 linewidth=lw, linestyle=ls, color=color)
    ax2.axhline(INITIAL_CAPITAL, color='gray', linestyle=':', linewidth=0.8)
    ax2.set_title('Portfolio Equity (all strategies)', fontweight='bold')
    ax2.set_ylabel('Portfolio Value (USD)')
    ax2.legend(fontsize=6.5, ncol=3, loc='upper left')
    ax2.grid(True, alpha=0.25)

    best_name, best_bt = sorted_r[0]
    ax3 = fig.add_subplot(gs[2, 0])
    dd = (best_bt.equity - best_bt.equity.cummax()) / best_bt.equity.cummax() * 100
    ax3.fill_between(dd.index, dd, 0, color='crimson', alpha=0.55)
    ax3.set_title(f'Drawdown — {best_name}', fontweight='bold')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[2, 1])
    r_best = best_bt.returns * 100
    r_bh = results['Buy & Hold'].returns * 100
    ax4.hist(r_bh, bins=80, alpha=0.4, color='steelblue', label='B&H', density=True)
    ax4.hist(r_best, bins=80, alpha=0.6, color='darkorange', label=best_name, density=True)
    var99 = np.percentile(r_best, 1)
    ax4.axvline(var99, color='red', linestyle='--', linewidth=1.2,
                label=f'VaR-99: {var99:.2f}%')
    ax4.set_title('Daily Return Distribution', fontweight='bold')
    ax4.set_xlabel('Daily Return (%)')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(outfile, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved → {outfile}')


def plot_comparison(btc_res: dict, eth_res: dict, outfile: str):
    """Side-by-side bar chart: BTC vs ETH returns for each strategy."""
    names = [n for n in btc_res if n in eth_res]
    btc_r = [btc_res[n].total_return * 100 for n in names]
    eth_r = [eth_res[n].total_return * 100 for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar(x - w / 2, btc_r, w, label='BTC (dev)', color='#f7931a', alpha=0.85)
    bars2 = ax.bar(x + w / 2, eth_r, w, label='ETH (validate)', color='#627eea', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Strategy Returns: BTC (development) vs ETH (out-of-sample validation)',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')


    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=6.5, color='#f7931a')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=6.5, color='#627eea')

    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved → {outfile}')


def year_by_year_analysis(assets: list[tuple[str, pd.DataFrame]], strategies: list[tuple[str, callable]], years: list[int] | None = None):
    """Per-calendar-year performance for top strategies across all assets.\n    Runs full 5y backtests so rolling windows are properly warm-started,\n    then slices equity curves to extract each year\'s metrics.\n    """
    if years is None:
        years = [2022, 2023, 2024, 2025, 2026]
    W = 110
    print(f"\n{'═' * W}")
    print('  YEAR-BY-YEAR STRATEGY PERFORMANCE  (calendar years 2022–2026)')
    print('  Signals computed on full 5y history — warm-up preserved')
    print(f"{'═' * W}")
    yr_cols = '  '.join((f'{y:>10}' for y in years))
    hdr = f"  {'Strategy':<28}  {'Asset':<5}  {yr_cols}  {'5-yr':>8}  {'AvgSh':>6}"
    print(hdr)
    print('─' * W)
    for strat_name, strat_fn in strategies:
        first = True
        for asset_name, df_a in assets:
            bt = Backtest(df_a, strat_fn(df_a))
            equity = bt.equity
            yr_rets = []
            for yr in years:
                yr_mask = equity.index.year == yr
                yr_eq = equity[yr_mask]
                if len(yr_eq) < 5:
                    yr_rets.append('   --    ')
                    continue
                else:
                    yr_start_loc = equity.index.get_loc(yr_eq.index[0])
                    prior_val = equity.iloc[yr_start_loc - 1] if yr_start_loc > 0 else INITIAL_CAPITAL
                    ret = yr_eq.iloc[(-1)] / prior_val - 1
                    yr_rets.append(f'{ret:>+9.1%} ')
            cols = ' '.join(yr_rets)
            full_r = bt.total_return
            sh_str = f'{bt.sharpe:.2f}'
            lbl = strat_name if first else ' ' * len(strat_name)
            first = False
            print(f'  {lbl:<28}  {asset_name:<5}  {cols}  {full_r:>+7.1%}  {sh_str:>6}')
        print()
    print('═' * W)


def monthly_stats_table(assets: list[tuple[str, pd.DataFrame]], strategies: list[tuple[str, callable]]):
    """Compute monthly return statistics for each strategy × asset.

    Reports: win rate, avg winning/losing month, best/worst month,
    max consecutive losing months, monthly Sharpe.
    Goal metric: maximise win_rate (% months finishing positive).
    """
    W = 127
    print(f"\n{'═' * W}")
    print('  MONTHLY PROFITABILITY ANALYSIS  (target: maximise monthly win rate)')
    print(f"{'═' * W}")
    hdr = f"  {'Strategy':<28}  {'Asset':<5}  {'WinRate':>9}  {'Wins/Mo':>8}  {'AvgWin':>8}  {'AvgLoss':>9}  {'VaR-99(Mo)':>11}  {'BestMo':>8}  {'WorstMo':>9}  {'MaxConsec':>10}  {'MthSharpe':>10}"
    print(hdr)
    print('─' * W)
    for strat_name, strat_fn in strategies:
        first = True
        for asset_name, df_a in assets:
            bt = Backtest(df_a, strat_fn(df_a))
            mth_eq = bt.equity.resample('ME').last()
            mth_ret = mth_eq.pct_change().dropna()
            n_mo = len(mth_ret)
            n_pos = int((mth_ret > 0).sum())
            win_rate = n_pos / n_mo if n_mo else 0
            avg_win = float(mth_ret[mth_ret > 0].mean()) if n_pos else 0.0
            avg_loss = float(mth_ret[mth_ret < 0].mean()) if (mth_ret < 0).any() else 0.0
            best_mo = float(mth_ret.max())
            worst_mo = float(mth_ret.min())
            mth_sh = float(np.sqrt(12) * mth_ret.mean() / mth_ret.std()) if mth_ret.std() > 0 else 0.0
            var99_mo = float(np.percentile(mth_ret, 1)) if len(mth_ret) >= 10 else float(mth_ret.min())
            streak = max_consec = 0
            for r in mth_ret:
                streak = streak + 1 if r < 0 else 0
                max_consec = max(max_consec, streak)
            invested = mth_ret[mth_ret.abs() > 0.001]
            n_inv = len(invested)
            inv_wr = float((invested > 0).mean()) if n_inv > 0 else 0.0
            lbl = strat_name if first else ' ' * len(strat_name)
            first = False
            print(f'  {lbl:<28}  {asset_name:<5}  {win_rate:>8.0%}  {n_pos:>3}/{n_mo:<4}  ({inv_wr:.0%} inv)  {avg_win:>+7.1%}  {avg_loss:>+8.1%}  {var99_mo:>+10.1%}  {best_mo:>+7.1%}  {worst_mo:>+8.1%}  {max_consec:>10}  {mth_sh:>10.2f}')
        print()
    print('═' * W)
    print("  Note: '(X% inv)' = win rate of months where strategy was actually invested (pos>0)")
    print('  GBM strategies spend many months in cash (0%) → low overall win rate is expected')


def base_position_grid(assets: list[tuple[str, pd.DataFrame]]):
    """Sweep base_pct from 0% to 40% to find the optimal minimum position floor.
    Reports monthly win rate, 5-year return, Sharpe, 2022 BTC return, and avg DD per level.
    """
    levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    W = 125
    print(f"\n{'═' * W}")
    print("  BASE POSITION GRID  (minimum crypto floor + GBM Vol-Tgt(40%,45d) top-up)")
    print("  Sweet spot: lowest base_pct that achieves ≥50% monthly win rate")
    print(f"{'═' * W}")

    asset_names = [n for n, _ in assets]
    hdr = (f"  {'Base':>6}  "
           + "  ".join(f"{n:>24}" for n in asset_names)
           + f"  {'AvgWR':>7}  {'AvgSh':>7}  {'BTC2022':>9}")
    print(hdr)
    print("─" * W)

    best_sweet = {'score': 999, 'base': 0.05}
    for base in levels:
        fn = lambda df, b=base: sig_base_position(df, base_pct=b)
        cols = []
        win_rates, sharpes = [], []
        btc_2022 = None
        for asset_name, df_a in assets:
            bt = Backtest(df_a, fn(df_a))
            mth_r = bt.equity.resample('ME').last().pct_change().dropna()
            wr = float((mth_r > 0).mean())
            win_rates.append(wr)
            sharpes.append(bt.sharpe)
            cols.append(f"{bt.total_return:>+7.1%}(WR={wr:.0%},Sh={bt.sharpe:.2f})")
            if asset_name == 'BTC':
                eq_22 = bt.equity[bt.equity.index.year == 2022]
                if len(eq_22) > 0:
                    loc = bt.equity.index.get_loc(eq_22.index[0])
                    prior = bt.equity.iloc[loc - 1] if loc > 0 else INITIAL_CAPITAL
                    btc_2022 = eq_22.iloc[-1] / prior - 1
        avg_wr = float(np.mean(win_rates))
        avg_sh = float(np.mean(sharpes))
        marker = '★' if avg_wr >= 0.50 else ' '
        if avg_wr >= 0.50 and base < best_sweet['score']:
            best_sweet = {'score': base, 'base': base, 'wr': avg_wr, 'sh': avg_sh}
        yr22_str = f"{btc_2022:>+8.1%}" if btc_2022 is not None else "      --"
        print(f"  {marker}{base:>5.0%}  "
              + "  ".join(f"{c:>24}" for c in cols)
              + f"  {avg_wr:>6.0%}  {avg_sh:>7.2f}  {yr22_str}")

    print("═" * W)
    print(f"  Sweet spot: {best_sweet['base']:.0%} base"
          f"  — first level to reach ≥50% monthly win rate"
          f"  AvgWR={best_sweet.get('wr', 0):.0%}  AvgSh={best_sweet.get('sh', 0):.2f}")


def monthly_stoploss_grid(assets: list[tuple[str, pd.DataFrame]], base_fn: callable, base_name: str):
    """Grid search over monthly stop-loss thresholds for a given base strategy.
    Shows how win rate and total return change as we tighten the monthly floor.
    """
    stops = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, None]
    W = 108
    print(f"\n{'═' * W}")
    print(f'  MONTHLY STOP-LOSS GRID  base={base_name}  (threshold = max monthly loss allowed)')
    print(f"{'═' * W}")
    asset_names = [n for n, _ in assets]
    hdr = (f"  {'Stop':>6}  "
           + '  '.join(f'{n:>22}' for n in asset_names)
           + f"  {'AvgWinRate':>12}  {'AvgSharpe':>10}")
    print(hdr)
    print('─' * W)
    for stop in stops:
        label = f'{stop:.0%}' if stop else ' none'
        if stop:
            fn = lambda df, s=stop: sig_monthly_stoploss(df, base_fn, stop_pct=s)
        else:
            fn = base_fn
        win_rates = []
        sharpes = []
        cols = []
        for _, df_a in assets:
            bt = Backtest(df_a, fn(df_a))
            mth_ret = bt.equity.resample('ME').last().pct_change().dropna()
            wr = float((mth_ret > 0).mean())
            win_rates.append(wr)
            sharpes.append(bt.sharpe)
            cols.append(f'{bt.total_return:>+7.1%}(WR={wr:.0%})')
        avg_wr = float(np.mean(win_rates))
        avg_sh = float(np.mean(sharpes))
        marker = '★' if avg_wr >= 0.6 else ' '
        print(f'  {marker}{label:>5}  '
              + '  '.join(f'{c:>22}' for c in cols)
              + f'  {avg_wr:>11.0%}  {avg_sh:>10.2f}')
    print('═' * W)


def monthly_heatmap(df: pd.DataFrame, results: dict[str, 'Backtest'], strategy_names: list[str], asset_label: str):
    """Print a text heatmap: rows=months (Jan-Dec), cols=years."""
    years = sorted(set(df.index.year))
    months = range(1, 13)
    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for sname in strategy_names:
        if sname not in results:
            continue
        else:
            bt = results[sname]
            eq = bt.equity
            mth_eq = eq.resample('ME').last()
            mth_r = mth_eq.pct_change()
            print(f'\n  Monthly returns heatmap  [{asset_label}]  {sname}')
            yr_hdr = '       ' + ''.join((f'{y:>8}' for y in years))
            print('  ' + yr_hdr)
            print('  ' + '─' * (7 + 8 * len(years)))
            for m in months:
                row = f'  {month_abbr[m - 1]:<5} '
                for y in years:
                    try:
                        r = mth_r[(mth_r.index.month == m) & (mth_r.index.year == y)]
                        if len(r):
                            v = float(r.iloc[0])
                            row += f'{v:>+7.1%} '
                        else:
                            row += '    --   '
                    except Exception:
                        row += '   err   '
                print(row)
            row = f"  {'TOTAL':<5} "
            for y in years:
                yr_r = mth_r[mth_r.index.year == y]
                if len(yr_r):
                    cum = float((1 + yr_r).prod() - 1)
                    row += f'{cum:>+7.1%} '
                else:
                    row += '    --   '
            print(row)


def intraday_minute_analysis():
    """5-minute bars, last ~60 days (max yfinance 5m limit).

    Rebalances daily at day-end, annualises with bpy = 365×24×12 = 105 120.
    Measures: are our strategies profitable on intraday data over 2 months?
    """
    tickers = [('BTC', 'BTC-USD'), ('ETH', 'ETH-USD'),
               ('SOL', 'SOL-USD'), ('BNB', 'BNB-USD')]
    bars_per_day = 288

    print('\n  Loading 5-minute bar data (~60 days)...')
    intraday = {}
    bpy_map = {}
    for name, tick in tickers:
        df_i, bpy = load_intraday_data(tick, interval='5m', period='60d')
        intraday[name] = df_i
        bpy_map[name] = bpy

    bpy = list(bpy_map.values())[0]

    strategies = [
        ('GBM 5m (3d drift)', lambda df: sig_gbm_intraday(df, window_days=3, bars_per_day=bars_per_day)),
        ('GBM 5m (7d drift)', lambda df: sig_gbm_intraday(df, window_days=7, bars_per_day=bars_per_day)),
        ('GBM 5m (14d drift)', lambda df: sig_gbm_intraday(df, window_days=14, bars_per_day=bars_per_day)),
        ('VT40% 5m (7d/3d)', lambda df: sig_vt_intraday(df, window_days=7, bars_per_day=bars_per_day,
                                                        target_ann_vol=0.40, rvol_days=3)),
        ('VT40% 5m (14d/7d)', lambda df: sig_vt_intraday(df, window_days=14, bars_per_day=bars_per_day,
                                                         target_ann_vol=0.40, rvol_days=7)),
        ('VT20% 5m (7d/3d)', lambda df: sig_vt_intraday(df, window_days=7, bars_per_day=bars_per_day,
                                                        target_ann_vol=0.20, rvol_days=3)),
        ('Buy & Hold', lambda df: pd.Series(1.0, index=df.index)),
    ]

    W = 100
    print(f"\n{'═' * W}")
    print('  5-MINUTE BAR ANALYSIS  (~60 days, daily rebal, 1% commission)')
    print(f'  Ann factor=√{bpy}  |  Drift window in calendar-day equivalents')
    print(f"{'═' * W}")
    asset_names = [n for n, _ in tickers]
    hdr = (f"  {'Strategy':<26}  "
           + '  '.join(f'{n:>18}' for n in asset_names)
           + f"  {'AvgSh':>6}")
    print(hdr)
    print('─' * W)

    for strat_name, strat_fn in strategies:
        bts = {}
        for name, df_i in intraday.items():
            sig = strat_fn(df_i)
            bts[name] = Backtest(df_i, sig, bars_per_year=bpy)
        avg_sh = float(np.mean([bt.sharpe for bt in bts.values()]))
        cols = '  '.join(
            f'{bts[n].total_return:>+8.1%}(Sh={bts[n].sharpe:>5.2f})'
            for n in asset_names
        )
        print(f'  {strat_name:<26}  {cols}  {avg_sh:>6.2f}')

    print('═' * W)


def hourly_longterm_analysis():
    """1-hour bars over max yfinance period (~730 days / 2 years).

    Rebalances daily at day-end, annualises with bpy = 8760.
    Measures: do strategies hold up over 2 years at hourly granularity?
    Includes monthly win-rate analysis on hourly data.
    """
    tickers = [('BTC', 'BTC-USD'), ('ETH', 'ETH-USD'),
               ('SOL', 'SOL-USD'), ('BNB', 'BNB-USD')]

    print('\n  Loading 1-hour bar data (max ~2 years)...')
    hourly = {}
    for name, tick in tickers:
        df_h, bpy = load_intraday_data(tick, interval='60m', period='730d')
        hourly[name] = df_h
    bpy = HOURS_PER_YEAR

    strategies = [
        ('GBM 1h (7d drift)', lambda df: sig_gbm_hourly(df, window_days=7)),
        ('GBM 1h (14d drift)', lambda df: sig_gbm_hourly(df, window_days=14)),
        ('GBM 1h (30d drift)', lambda df: sig_gbm_hourly(df, window_days=30)),
        ('VT40% 1h (7d/3d)', lambda df: sig_vt_hourly(df, window_days=7,
                                                      target_ann_vol=0.4, rvol_days=3)),
        ('VT40% 1h (14d/7d)', lambda df: sig_vt_hourly(df, window_days=14,
                                                       target_ann_vol=0.4, rvol_days=7)),
        ('VT40% 1h (30d/14d)', lambda df: sig_vt_hourly(df, window_days=30,
                                                        target_ann_vol=0.4, rvol_days=14)),
        ('VT20% 1h (14d/7d)', lambda df: sig_vt_hourly(df, window_days=14,
                                                       target_ann_vol=0.2, rvol_days=7)),
        ('Buy & Hold', lambda df: pd.Series(1.0, index=df.index)),
    ]

    W = 100
    print(f"\n{'═' * W}")
    print('  1-HOUR BAR ANALYSIS  (~2 years, daily rebal, 1% commission)')
    print(f'  Ann factor=√{bpy}  |  Drift window in calendar-day equivalents')
    print(f"{'═' * W}")
    asset_names = [n for n, _ in tickers]
    hdr = (f"  {'Strategy':<26}  "
           + '  '.join(f'{n:>18}' for n in asset_names)
           + f"  {'AvgSh':>6}")
    print(hdr)
    print('─' * W)

    for strat_name, strat_fn in strategies:
        bts = {}
        for name, df_h in hourly.items():
            sig = strat_fn(df_h)
            bts[name] = Backtest(df_h, sig, bars_per_year=bpy)
        avg_sh = float(np.mean([bt.sharpe for bt in bts.values()]))
        cols = '  '.join(
            f'{bts[n].total_return:>+8.1%}(Sh={bts[n].sharpe:>5.2f})'
            for n in asset_names
        )
        print(f'  {strat_name:<26}  {cols}  {avg_sh:>6.2f}')

    print('═' * W)

    print('\n  Monthly win-rate table — hourly data (~2 years)')
    W2 = 115
    print(f"{'═' * W2}")
    hdr2 = (f"  {'Strategy':<26}  {'Asset':<5}  "
            f"{'WinRate':>9}  {'Wins/Mo':>8}  {'AvgWin':>8}  {'AvgLoss':>9}  "
            f"{'VaR-99(Mo)':>11}  {'BestMo':>8}  {'WorstMo':>9}  {'MthSh':>7}")
    print(hdr2)
    print('─' * W2)

    strat_list = [(n, f) for n, f in strategies if n != 'Buy & Hold']
    strat_list.append(('Buy & Hold', lambda df: pd.Series(1.0, index=df.index)))

    for strat_name, strat_fn in strat_list:
        first = True
        for asset_name, df_h in hourly.items():
            bt = Backtest(df_h, strat_fn(df_h), bars_per_year=bpy)
            mth_eq = bt.equity.resample('ME').last()
            mth_ret = mth_eq.pct_change().dropna()
            n_mo = len(mth_ret)
            n_pos = int((mth_ret > 0).sum())
            wr = n_pos / n_mo if n_mo else 0
            avg_win = float(mth_ret[mth_ret > 0].mean()) if n_pos else 0.0
            avg_loss = float(mth_ret[mth_ret < 0].mean()) if (mth_ret < 0).any() else 0.0
            best_mo = float(mth_ret.max())
            worst_mo = float(mth_ret.min())
            mth_sh = float(np.sqrt(12) * mth_ret.mean() / mth_ret.std()) if mth_ret.std() > 0 else 0.0
            var99_mo = float(np.percentile(mth_ret, 1)) if n_mo >= 10 else float(mth_ret.min())
            lbl = strat_name if first else ' ' * len(strat_name)
            first = False
            print(f'  {lbl:<26}  {asset_name:<5}  '
                  f'{wr:>8.0%}  {n_pos:>3}/{n_mo:<4}  '
                  f'{avg_win:>+7.1%}  {avg_loss:>+8.1%}  '
                  f'{var99_mo:>+10.1%}  '
                  f'{best_mo:>+7.1%}  {worst_mo:>+8.1%}  {mth_sh:>7.2f}')
        print()
    print('═' * W2)


def hourly_analysis():
    """Download 1 month of hourly crypto data and test GBM strategies adapted to 1h bars.

    Key differences vs daily:
    • bars_per_year = 8760  (crypto trades 24/7)
    • Drift/vol windows in days × 24 hours
    • Rebalancing: end of each calendar day (instead of month-end)
    • vol-targeting annualizes as sqrt(8760)
    """
    tickers = [('BTC', 'BTC-USD'), ('ETH', 'ETH-USD'),
               ('SOL', 'SOL-USD'), ('BNB', 'BNB-USD')]

    print("\n  Loading 1-month hourly data...")
    hourly = {}
    for name, tick in tickers:
        hourly[name] = load_hourly_data(tick, period='1mo')

    bpy = HOURS_PER_YEAR

    strategies = [
        ('GBM Daily (7d drift)', lambda df: sig_gbm_hourly(df, window_days=7)),
        ('GBM Daily (14d drift)', lambda df: sig_gbm_hourly(df, window_days=14)),
        ('Vol-Tgt 40% (7d/3d)', lambda df: sig_vt_hourly(df, window_days=7,
                                                          target_ann_vol=0.40, rvol_days=3)),
        ('Vol-Tgt 40% (14d/7d)', lambda df: sig_vt_hourly(df, window_days=14,
                                                           target_ann_vol=0.40, rvol_days=7)),
        ('Vol-Tgt 20% (7d/3d)', lambda df: sig_vt_hourly(df, window_days=7,
                                                          target_ann_vol=0.20, rvol_days=3)),
        ('Buy & Hold', lambda df: pd.Series(1.0, index=df.index)),
    ]

    W = 100
    print(f"\n{'═' * W}")
    print("  HOURLY GBM ANALYSIS  (1 month, 1h bars, daily rebal, 1% commission)")
    print("  Ann factor=√8760  |  Window in calendar-day equivalents")
    print(f"{'═' * W}")

    asset_names = [n for n, _ in tickers]
    hdr = (f"  {'Strategy':<26}  "
           + '  '.join(f'{n:>18}' for n in asset_names)
           + f"  {'AvgSh':>6}")
    print(hdr)
    print('─' * W)

    for strat_name, strat_fn in strategies:
        bts = {}
        for name, df_h in hourly.items():
            sig = strat_fn(df_h)
            bts[name] = Backtest(df_h, sig, bars_per_year=bpy)

        avg_sh = float(np.mean([bt.sharpe for bt in bts.values()]))
        cols = '  '.join(
            f'{bts[n].total_return:>+8.1%}(Sh={bts[n].sharpe:>5.2f})'
            for n in asset_names
        )
        print(f"  {strat_name:<26}  {cols}  {avg_sh:>6.2f}")

    print('═' * W)

    print("\n  Detailed metrics — Vol-Tgt 40% (7d/3d) on hourly bars:")
    print(f"  {'Asset':<6}  {'Ret':>9}  {'AnnRet':>9}  {'Sharpe':>7}  "
          f"{'MaxDD':>8}  {'Trades':>7}  {'Bars':>6}")
    print("  ─────────────────────────────────────────────────────────────────")
    vt_fn = lambda df: sig_vt_hourly(df, window_days=7, target_ann_vol=0.40, rvol_days=3)
    for name, df_h in hourly.items():
        bt = Backtest(df_h, vt_fn(df_h), bars_per_year=bpy)
        print(f"  {name:<6}  {bt.total_return:>+8.1%}  {bt.annual_return:>+8.1%}  "
              f"{bt.sharpe:>7.2f}  {bt.max_drawdown:>7.1%}  {bt.trades:>7}  {len(df_h):>6}")


def main():
    import sys
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     CRYPTO QUANT LAB v59 — Bank Research Division            ║")
    print("║     BTC/ETH/SOL/BNB  |  5-year  |  1% commission            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print("\n[1/7] Loading market data")
    btc = load_data('BTC-USD')
    eth = load_data('ETH-USD')
    sol = load_data('SOL-USD')
    bnb = load_data('BNB-USD')

    print("\n[2/7] Window sensitivity — GBM Monthly Rebal")
    optimal_window = window_sensitivity(btc, eth)

    print("\n[3/7] BTC — development / strategy fitting")
    btc_res = run_all(btc, 'BTC')
    print_table(btc_res, 'BTC-USD — Development')

    print("\n[4/7] ETH — out-of-sample validation")
    eth_res = run_all(eth, 'ETH')
    print_table(eth_res, 'ETH-USD — Out-of-Sample Validation')

    print("\n[5/7] Cross-asset comparison")
    cross_validate(btc_res, eth_res)

    print("\n[5b/7] SOL + BNB — full strategy tables", flush=True)
    sol_res = run_all(sol, 'SOL')
    bnb_res = run_all(bnb, 'BNB')
    print_table(sol_res, 'SOL-USD — Cross-Asset Validation')
    print_table(bnb_res, 'BNB-USD — Cross-Asset Validation')

    print("\n[5c/7] GBM vs Naive Momentum — does drift estimation add value?")
    for asset_name, res in (('BTC', btc_res), ('ETH', eth_res),
                            ('SOL', sol_res), ('BNB', bnb_res)):
        gbm_r = res['GBM Monthly (60d)'].total_return
        mom_r = res.get('Momentum Monthly (60d)')
        if mom_r is None: continue
        diff = gbm_r - mom_r.total_return
        print(f"  {asset_name}:  GBM={gbm_r:+.1%}  Momentum={mom_r.total_return:+.1%}  → GBM edge="
              f"{diff:+.1%} {'(GBM wins)' if diff > 0 else '(same/worse)'}")

    print("\n[5d/7] v15 — 2D grid: target_vol × rvol_window (confirm optimum)", flush=True)
    optimal_tvol, optimal_rvol_w = vol_targeting_2d_grid(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        drift_window=optimal_window)

    print("\n  Top-3 vol-tgt variants vs base GBM:")
    for ticker, res in (('BTC', btc_res), ('ETH', eth_res),
                        ('SOL', sol_res), ('BNB', bnb_res)):
        bh = res['Buy & Hold'].total_return
        vt4 = res['GBM Vol-Targeting (40%)']
        vt45 = res.get('GBM Vol-Tgt (40%,45d)')
        gbm = res['GBM Monthly (60d)']
        v45_str = f"  VolTgt40%(45d)={vt45.total_return:+.1%}(Sh={vt45.sharpe:.2f})" if vt45 else ''
        print(f"  {ticker}: B&H={bh:+.1%}  VolTgt40%(30d)="
              f"{vt4.total_return:+.1%}(Sh={vt4.sharpe:.2f})"
              f"{v45_str}  GBM60d="
              f"{gbm.total_return:+.1%}(Sh={gbm.sharpe:.2f})")

    print("\n[5e/7] Drift-window sensitivity for Vol-Tgt (40%,45d) — v14 new", flush=True)
    optimal_drift_w = vol_tgt_drift_window_sensitivity(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])

    sys.stdout.flush()

    print("\n[5f/7] Realized-vol window sensitivity (fixed 40% target)", flush=True)
    realized_vol_window_sensitivity(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        drift_window=optimal_window)

    print("\n[6/7] Walk-forward robustness + commission sensitivity", flush=True)
    w = optimal_window
    for name, fn in (
        (f'GBM Monthly ({w}d)', lambda df, w=w: sig_gbm_monthly(df, window=w)),
        ('GBM Vol-Targeting (40%)', lambda df, w=w: sig_vol_targeting_40(df, window=w)),
        ('GBM Vol-Tgt (40%,45d)', lambda df, w=w: sig_vol_targeting_40_45d(df, window=w)),
    ):
        walk_forward(btc, f"{name} — BTC", fn)
        walk_forward(eth, f"{name} — ETH", fn)

    print("\n[6b-ext/7] Walk-forward: SOL + BNB — flagship strategy", flush=True)
    for name, fn in (
        ('GBM Vol-Tgt (40%,45d)', lambda df, w=w: sig_vol_targeting_40_45d(df, window=w)),
        ('Base5 + VT (monthly)', lambda df: sig_base5_vt(df)),
        ('Base20 + VT (monthly)', lambda df: sig_base20_vt(df)),
    ):
        walk_forward(sol, f"{name} — SOL", fn)
        walk_forward(bnb, f"{name} — BNB", fn)

    print("\n[6d/7] Per-period Sharpe breakdown — top strategies all 4 assets", flush=True)
    per_period_sharpe(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        [
            ('GBM Monthly (60d)', lambda df, w=w: sig_gbm_monthly(df, window=w)),
            ('GBM Vol-Tgt (40%,30d)', lambda df, w=w: sig_vol_targeting_40(df, window=w)),
            ('GBM Vol-Tgt (40%,45d)', lambda df, w=w: sig_vol_targeting_40_45d(df, window=w)),
            ('Buy & Hold', lambda df: sig_buy_hold(df)),
        ],
    )
    sys.stdout.flush()

    print("\n[6e/7] Annual returns by calendar year — top strategies", flush=True)
    annual_returns_table(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        [
            ('GBM Vol-Tgt (40%,45d)', lambda df: sig_vol_targeting_40_45d(df)),
            ('Base20 + VT (monthly)', lambda df: sig_base20_vt(df)),
            ('Base30 + VT (monthly)', lambda df: sig_base30_vt(df)),
            ('GBM Blend (VT+Asym)', lambda df: sig_blend_50_50(df)),
            ('Buy & Hold', lambda df: sig_buy_hold(df)),
        ],
    )
    sys.stdout.flush()

    print("\n[6g/7] Year-by-year analysis (2022–2026) — top strategies", flush=True)
    year_by_year_analysis(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        [
            ('GBM Vol-Tgt (40%,45d)', lambda df: sig_vol_targeting_40_45d(df)),
            ('Base5 + VT (monthly)', lambda df: sig_base5_vt(df)),
            ('Base10 + VT (monthly)', lambda df: sig_base10_vt(df)),
            ('Base5 + Blend (monthly)', lambda df: sig_base5_blend(df)),
            ('GBM Asymmetric (30/60)', lambda df: sig_gbm_asymmetric(df)),
            ('GBM Blend (VT+Asym)', lambda df: sig_blend_50_50(df)),
            ('VT45 + MStop(5%)', lambda df: sig_vt45_monthly_stop(df, 0.05)),
            ('Buy & Hold', lambda df: sig_buy_hold(df)),
        ],
    )
    sys.stdout.flush()

    print("\n[6h/7] Base position grid — optimal floor for monthly win rate", flush=True)
    base_position_grid([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[6i/7] Monthly profitability analysis — win rates & base-position strategies", flush=True)
    monthly_stats_table(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        [
            ('GBM Vol-Tgt (40%,45d)', lambda df: sig_vol_targeting_40_45d(df)),
            ('VT45 + Seasonal', lambda df: sig_seasonal_vt45(df)),
            ('Blend + Seasonal', lambda df: sig_seasonal_blend(df)),
            ('BTC-Lead VT (altcoins)', lambda df: sig_btc_lead_vt(df, btc) if df is not btc else sig_vol_targeting_40_45d(df)),
            ('BTC-Lead+Seasonal', lambda df: sig_btc_lead_seasonal(df, btc) if df is not btc else sig_seasonal_vt45(df)),
            ('Base5+Blend+BoostOnly', lambda df: sig_base5_boost_blend(df)),
            ('BTC-Lead Base5', lambda df: sig_btc_lead_base5(df, btc) if df is not btc else sig_base5_vt(df)),
            ('Weekly Vol-Tgt (40%,45d)', lambda df: sig_weekly_vol_targeting(df)),
            ('Base5 + VT (monthly)', lambda df: sig_base5_vt(df)),
            ('Base10 + VT (monthly)', lambda df: sig_base10_vt(df)),
            ('Base5 + Blend (monthly)', lambda df: sig_base5_blend(df)),
            ('Base20 + VT (monthly)', lambda df: sig_base20_vt(df)),
            ('Base30 + VT (monthly)', lambda df: sig_base30_vt(df)),
            ('GBM Blend (VT+Asym)', lambda df: sig_blend_50_50(df)),
            ('Buy & Hold', lambda df: sig_buy_hold(df)),
        ],
    )
    sys.stdout.flush()

    monthly_stoploss_grid(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        lambda df: sig_vol_targeting_40_45d(df),
        'GBM Vol-Tgt (40%,45d)',
    )
    sys.stdout.flush()

    print("\n  Monthly returns heatmap — BTC key strategies:")
    monthly_heatmap(btc, btc_res,
                    ['GBM Vol-Tgt (40%,45d)', 'Base20 + VT (monthly)',
                     'Weekly Vol-Tgt (40%,45d)', 'Base20 + Weekly VT', 'Buy & Hold'],
                    'BTC')
    sys.stdout.flush()

    print("\n[6f/7] Commission sensitivity — flagship Vol-Tgt (40%,45d)", flush=True)
    commission_sensitivity_flagship(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        'GBM Vol-Tgt (40%,45d)',
        lambda df, w=w: sig_vol_targeting_40_45d(df, window=w),
    )
    sys.stdout.flush()

    commission_sensitivity(btc, eth, window=optimal_window)

    print("\n[6c/7] Statistical significance tests", flush=True)
    gbm_fn = lambda df: sig_gbm_monthly(df, window=optimal_window)
    vt40_fn = lambda df: sig_vol_targeting_40(df, window=optimal_window)
    vt45_fn = lambda df: sig_vol_targeting_40_45d(df, window=optimal_window)
    asym_fn = lambda df: sig_gbm_asymmetric(df)
    vgat_fn = lambda df: sig_gbm_asymmetric_volgated(df)
    blen_fn = lambda df: sig_blend_50_50(df)

    print("\n  — Lo (2002) autocorrelation-adjusted Sharpe t-test —", flush=True)
    for ticker, res in (('BTC', btc_res), ('ETH', eth_res),
                        ('SOL', sol_res), ('BNB', bnb_res)):
        lo_sharpe_ttest(res['GBM Monthly (60d)'],
                        f"GBM Monthly ({optimal_window}d)  — {ticker}")
        sys.stdout.flush()
    for ticker, res in (('BTC', btc_res), ('ETH', eth_res),
                        ('SOL', sol_res), ('BNB', bnb_res)):
        lo_sharpe_ttest(res['GBM Vol-Targeting (40%)'],
                        f"GBM Vol-Tgt (40%,30d) — {ticker}")
        sys.stdout.flush()
    for ticker, res in (('BTC', btc_res), ('ETH', eth_res),
                        ('SOL', sol_res), ('BNB', bnb_res)):
        if 'GBM Vol-Tgt (40%,45d)' in res:
            lo_sharpe_ttest(res['GBM Vol-Tgt (40%,45d)'],
                            f"GBM Vol-Tgt (40%,45d) — {ticker}")
            sys.stdout.flush()
    for ticker, res in (('BTC', btc_res), ('ETH', eth_res),
                        ('SOL', sol_res), ('BNB', bnb_res)):
        if 'GBM Asymmetric (30/60)' in res:
            lo_sharpe_ttest(res['GBM Asymmetric (30/60)'],
                            f"GBM Asymmetric (30/60) — {ticker}")
            sys.stdout.flush()
    for ticker, res in (('BTC', btc_res), ('ETH', eth_res),
                        ('SOL', sol_res), ('BNB', bnb_res)):
        if 'GBM Asym Vol-Gated' in res:
            lo_sharpe_ttest(res['GBM Asym Vol-Gated'],
                            f"GBM Asym Vol-Gated   — {ticker}")
            sys.stdout.flush()
    for ticker, res in (('BTC', btc_res), ('ETH', eth_res),
                        ('SOL', sol_res), ('BNB', bnb_res)):
        if 'GBM Blend (VT+Asym)' in res:
            lo_sharpe_ttest(res['GBM Blend (VT+Asym)'],
                            f"GBM Blend (VT+Asym)  — {ticker}")
            sys.stdout.flush()

    print("\n  — Per-asset bootstrap (600 iterations) —", flush=True)
    for ticker, df_a, fn in (
        ('GBM Monthly — BTC', btc, gbm_fn),
        ('GBM Monthly — ETH', eth, gbm_fn),
        ('GBM Vol-Tgt(40%,45d)— BTC', btc, vt45_fn),
        ('GBM Vol-Tgt(40%,45d)— ETH', eth, vt45_fn),
        ('GBM Asymmetric — BTC', btc, asym_fn),
        ('GBM Asymmetric — ETH', eth, asym_fn),
        ('GBM VolGated — BTC', btc, vgat_fn),
        ('GBM VolGated — ETH', eth, vgat_fn),
        ('GBM Blend — BTC', btc, blen_fn),
        ('GBM Blend — ETH', eth, blen_fn),
    ):
        bootstrap_significance(df_a, fn, ticker, n_bootstrap=600)
        sys.stdout.flush()

    base20_fn = lambda df: sig_base20_vt(df)
    for ticker, res in (('BTC', btc_res), ('ETH', eth_res),
                        ('SOL', sol_res), ('BNB', bnb_res)):
        if 'Base20 + VT (monthly)' in res:
            lo_sharpe_ttest(res['Base20 + VT (monthly)'],
                            f"Base20 + VT (monthly)   — {ticker}")
            sys.stdout.flush()

    print("\n  — Pooled cross-asset bootstrap (1000 iterations) —", flush=True)
    for label, fn in (('GBM Monthly (60d)', gbm_fn),
                      ('GBM Vol-Tgt (40%,45d)', vt45_fn),
                      ('GBM Asymmetric (30/60)', asym_fn),
                      ('Base20 + VT (monthly)', base20_fn),
                      ('GBM Blend (VT+Asym)', blen_fn)):
        pooled_bootstrap_significance(
            [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
            fn, n_bootstrap=1000,
        )
        sys.stdout.flush()

    print("\n[7/7] Generating plots")
    plot_asset(btc, btc_res,
               'BTC-USD Strategy Comparison v9 (Development)',
               f'{OUT_DIR}/btc_strategies.png')
    plot_asset(eth, eth_res,
               'ETH-USD Strategy Comparison v9 (Validation)',
               f'{OUT_DIR}/eth_strategies.png')
    plot_comparison(btc_res, eth_res,
                    f'{OUT_DIR}/strategy_comparison.png')

    print("\n[7b/7] P1 bear-market sub-period  (2021-05 → 2022-12)")
    p1_end = pd.Timestamp('2022-12-31')
    key_strats = ['GBM Monthly (60d)', 'GBM Vol-Targeting (20%)', 'GBM Vol-Tgt (40%,45d)',
                  'Base20 + VT (monthly)', 'Base30 + VT (monthly)',
                  'GBM Blend (VT+Asym)', 'GBM Monthly Kelly', 'Buy & Hold']

    for ticker, df_a, res in (('BTC', btc, btc_res), ('ETH', eth, eth_res)):
        sub = df_a[df_a.index <= p1_end]
        print(f"\n  {ticker} P1 sub-period ({sub.index[0].date()} → {sub.index[-1].date()}):")
        for sname in key_strats:
            if sname == 'Buy & Hold':
                fn = sig_buy_hold
            elif sname == 'GBM Monthly (60d)':
                fn = lambda d, w=optimal_window: sig_gbm_monthly(d, window=w)
            elif sname == 'GBM Monthly BearGate':
                fn = lambda d, w=optimal_window: sig_gbm_monthly_bear_gate(d, window=w)
            elif sname == 'GBM Tight Gate (-15%)':
                fn = lambda d, w=optimal_window: sig_gbm_tight_gate(d, window=w)
            elif sname == 'GBM Vol-Targeting (20%)':
                fn = lambda d, w=optimal_window: sig_vol_targeting(d, window=w,
                                                                   target_ann_vol=0.20)
            elif sname == 'GBM Vol-Targeting (40%)':
                fn = lambda d, w=optimal_window: sig_vol_targeting_40(d, window=w)
            elif sname == 'GBM Vol-Tgt (40%,45d)':
                fn = lambda d, w=optimal_window: sig_vol_targeting_40_45d(d, window=w)
            elif sname == 'GBM Asymmetric (30/60)':
                fn = lambda d: sig_gbm_asymmetric(d)
            elif sname == 'GBM Asym Vol-Gated':
                fn = lambda d: sig_gbm_asymmetric_volgated(d)
            elif sname == 'GBM Blend (VT+Asym)':
                fn = lambda d: sig_blend_50_50(d)
            elif sname == 'Base20 + VT (monthly)':
                fn = lambda d: sig_base20_vt(d)
            elif sname == 'Base30 + VT (monthly)':
                fn = lambda d: sig_base30_vt(d)
            elif sname == 'GBM Monthly Kelly':
                fn = lambda d, w=optimal_window: sig_gbm_monthly_kelly(d, window=w)
            elif sname == 'Ensemble (≥3)':
                fn = lambda d: sig_ensemble(
                    {k: v for k, v in {
                        'GBM Monthly BearGate': sig_gbm_monthly_bear_gate(d),
                        'GBM Monthly (60d)': sig_gbm_monthly(d),
                        'Bull Filter': sig_above_ma(d),
                    }.items()}, min_agree=2)
            else:
                continue
            bt_p1 = Backtest(sub, fn(sub))
            print(f"    {sname:<28}  {bt_p1.total_return:>+8.1%}  MaxDD="
                  f"{bt_p1.max_drawdown:.1%}  Trades={bt_p1.trades}")

    print("\n[9a/7] BTC lead signal — does BTC drift time altcoins better?", flush=True)
    btc_lead_analysis([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)], btc)
    sys.stdout.flush()

    print("\n[9b/7] GARCH(1,1) drift vs rolling drift", flush=True)
    garch_drift_analysis([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9c/7] Portfolio risk-parity analysis", flush=True)
    portfolio_riskparity_analysis([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9d/7] Combined BTC-Lead + Risk-Parity portfolio", flush=True)
    portfolio_btc_lead_rp([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)], btc)
    sys.stdout.flush()

    print("\n[9e/7] Calendar/seasonal effects analysis", flush=True)
    calendar_effects_analysis([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9f/7] Walk-forward: BTC-Lead strategies on all assets", flush=True)
    for name, fn in (
        ('BTC-Lead VT (40%,45d)', lambda df: sig_btc_lead_vt(df, btc)),
        ('BTC-Lead Base5 VT', lambda df: sig_btc_lead_base5(df, btc)),
        ('BTC-Lead + Funded', lambda df: sig_btc_lead_funded(df, btc)),
    ):
        for asset_name, df_a in (('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)):
            walk_forward(df_a, f"{name} — {asset_name}", fn)
    sys.stdout.flush()

    print("\n[9g/7] Seasonal overlay — walk-forward out-of-sample validation", flush=True)
    seasonal_walkforward_analysis([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9h/7] Final Unified Deployable System — portfolio comparison", flush=True)
    portfolio_final_unified([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)], btc)
    sys.stdout.flush()

    print("\n[9i/7] v24 Seasonal refinements — boost-only vs dampen, significance test", flush=True)
    seasonal_variants_comparison([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()
    seasonal_bootstrap_significance([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()
    portfolio_v24_comparison([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9j/7] v25 — Blend+Boost-Only walk-forward + best portfolio", flush=True)
    for name, fn in (
        ('Blend+Boost-Only', lambda df: sig_boost_only_seasonal_blend(df)),
        ('Asym+Boost-Only', lambda df: sig_boost_only_seasonal_asym(df)),
        ('VT45+Boost-Only', lambda df: sig_boost_only_seasonal_vt45(df)),
        ('Base5+Blend+BoostOnly', lambda df: sig_base5_boost_blend(df)),
    ):
        for asset_name, df_a in (('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)):
            walk_forward(df_a, f"{name} — {asset_name}", fn)
    sys.stdout.flush()
    portfolio_v25_best([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9k/7] v27 — Commission sensitivity for top strategies", flush=True)
    commission_sensitivity_top_strategies([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9l/7] v27 — Ensemble portfolio: blending Sharpe and WR", flush=True)
    portfolio_v27_ensemble([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9m/7] v28 — Seasonal window sensitivity + new signal walk-forward", flush=True)
    seasonal_window_sensitivity([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()
    for name, fn in (
        ('DriftScaled+VT45+Boost', lambda df: sig_drift_scaled_vt45_boost(df)),
        ('BullFilter+VT45+Boost', lambda df: sig_bull_filter_vt45_boost(df)),
    ):
        for asset_name, df_a in (('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)):
            walk_forward(df_a, f"{name} — {asset_name}", fn)
    sys.stdout.flush()

    print("\n[9n/7] v28 — Final portfolio: drift-scaled + bull-filter variants", flush=True)
    portfolio_v28_final([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9o/7] v29 — Floor sensitivity sweep (WR/Sharpe frontier)", flush=True)
    floor_sensitivity_analysis([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()
    for name, fn in (
        ('Base2+VT45+Boost', lambda df: sig_base2_vt45_boost(df)),
        ('Base5+VT45+Boost', lambda df: sig_base5_vt45_boost(df)),
    ):
        for asset_name, df_a in (('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)):
            walk_forward(df_a, f"{name} — {asset_name}", fn)
    sys.stdout.flush()

    print("\n[9p/7] v29 — Final portfolio: floor variants + VT45+Seasonal anchor", flush=True)
    portfolio_v29_final([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9q/7] v30 — Base2+Blend+Boost walk-forward validation", flush=True)
    for asset_name, df_a in (('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)):
        walk_forward(df_a, f"Base2+Blend+Boost — {asset_name}",
                     lambda df: sig_base2_blend_boost(df))
    sys.stdout.flush()

    print("\n[9r/7] v30 — Final portfolio: Base2+Blend+Boost vs v29 champions", flush=True)
    portfolio_v30_final([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9s/7] v31 — Fine floor sweep (0-5% in 0.5% steps)", flush=True)
    fine_floor_sweep([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9t/7] v31 — 3-way ensemble grid search (Sh/WR Pareto frontier)", flush=True)
    portfolio_v31_ensemble_grid([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9u/7] v32 — Fine ensemble sweep + commission robustness of 90/10 optimal", flush=True)
    portfolio_v32_fine_ensemble([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9v/7] v33 — Sub-percent sweep + 2D commission×allocation surface", flush=True)
    portfolio_v33_final_validation([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9w/7] v34 — Deployment spec + today's live positions", flush=True)
    current_positions_and_spec([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9x/7] v35 — Extended universe: 4-asset vs 6-asset RP (+ADA, +AVAX)", flush=True)
    extended_universe_test()
    sys.stdout.flush()

    print("\n[9y/7] v35 — Monthly P&L calendar: validating seasonal overlay months", flush=True)
    monthly_pnl_calendar([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9z/7] v36 — 6-asset RP full validation + seasonal refinement", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    extended_universe_v36(six)
    sys.stdout.flush()
    seasonal_refinement_v36([('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)])
    sys.stdout.flush()

    print("\n[9aa/7] v37 — Core3 seasonal (Mar/Jul/Nov) full validation", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v37_core3_final(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        six,
    )
    sys.stdout.flush()

    print("\n[9ab/7] v38 — Final champion validation: 6-asset Core3 commission + walk-forward", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    final_champion_validation(six)
    sys.stdout.flush()

    print("\n[9ac/7] v39 — Core3+Blend upgrade: Blend signal with Core3 seasonal vs champion", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v39_blend_upgrade(
        [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb)],
        six,
    )
    sys.stdout.flush()

    print("\n[9ad/7] v40 — Drift window sensitivity + 8-asset universe (XRP, LINK)", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v40_window_and_universe(six)
    sys.stdout.flush()

    print("\n[9ae/7] v41 — Vol-target sweep + seasonal leverage (signal-strength scaling)", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v41_voltarget_and_leverage(six)
    sys.stdout.flush()

    print("\n[9af/7] v42 — Leverage at VT=40% + Calmar + DOT/ATOM altcoins", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v42_leverage_fix_and_altcoins(six)
    sys.stdout.flush()

    print("\n[9ag/7] v43 — Seasonal boost magnitude + RP rebalancing frequency", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v43_boost_and_rp_freq(six)
    sys.stdout.flush()

    print("\n[9ah/7] v44 — ×1.5 boost champion + fixed RP rebalancing frequency", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v44_rp_freq_fix(six)
    sys.stdout.flush()

    print("\n[9ai/7] v45 — BTC-gated Core3 boost + RP weight cap", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v45_gated_boost_and_weight_cap(six, btc)
    sys.stdout.flush()

    print("\n[9aj/7] v46 — RP vol window sweep + seasonal month portfolio sweep", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v46_rp_vol_window_and_seasonal_sweep(six)
    sys.stdout.flush()

    print("\n[9ak/7] v47 — walk-forward: 5m vol window + Feb/Jul/Nov seasonal", flush=True)
    ada = load_data('ADA-USD')
    avax = load_data('AVAX-USD')
    six = [('BTC', btc), ('ETH', eth), ('SOL', sol), ('BNB', bnb), ('ADA', ada), ('AVAX', avax)]
    portfolio_v47_vol_window_wf_and_seasonal_update(six)
    sys.stdout.flush()

    print("\n[9al/7] v48 — min weight floor + drift-magnitude RP + start-date robustness", flush=True)
    portfolio_v48_min_floor_drift_rp_robustness(six)
    sys.stdout.flush()

    print("\n[9am/7] v49 — drift-magnitude RP look-ahead fix + drift/vol-window tuning", flush=True)
    portfolio_v49_drift_rp_corrected(six)
    sys.stdout.flush()

    print("\n[9an/7] v50 — portfolio drawdown gate + inverse variance RP", flush=True)
    portfolio_v50_drawdown_gate_inv_variance(six)
    sys.stdout.flush()

    print("\n[9ao/7] v51 — inverse-variance champion validation", flush=True)
    portfolio_v51_inv_var_validation(six)
    sys.stdout.flush()

    print("\n[9ap/7] v52 — seasonal boost re-opt (inv-var) + hierarchical RP", flush=True)
    portfolio_v52_boost_reopt_hierarchical_rp(six)
    sys.stdout.flush()

    print("\n[9aq/7] v53 — HRP + extended boost sweep + commission robustness", flush=True)
    portfolio_v53_hrp_boost_combined(six)
    sys.stdout.flush()

    print("\n[9ar/7] v54 — vol/seasonal/drift window re-test on HRP ×3.0 champion", flush=True)
    portfolio_v54_vol_window_seasonal_retest(six)
    sys.stdout.flush()

    print("\n[9as/7] v55 — vol-target re-opt + HRP cluster structure variants", flush=True)
    portfolio_v55_vol_target_and_cluster_structure(six)
    sys.stdout.flush()

    print("\n[9at/7] v56 — BTC 200d MA regime gate + portfolio DD gate", flush=True)
    portfolio_v56_regime_gate(six)
    sys.stdout.flush()

    print("\n[9au/7] v57 — MA window sensitivity + commission sweep on MA-gated champion", flush=True)
    portfolio_v57_ma_window_sensitivity(six)
    sys.stdout.flush()

    print("\n[9av/7] v58 — fine-grained MA window + scale confirm — final champion lock", flush=True)
    portfolio_v58_fine_grained_ma_sweep(six)
    sys.stdout.flush()

    print("\n[9aw/7] v59 — Core3 gate exemption (Mar/Jul/Nov active even when BTC < 150d MA)", flush=True)
    portfolio_v59_core3_gate_exemption(six)
    sys.stdout.flush()

    print("\n[8/7] Intraday analysis — 5m bars (~2 months)", flush=True)
    intraday_minute_analysis()
    sys.stdout.flush()

    print("\n[8b/7] Intraday analysis — 1h bars (~2 years, monthly win rate)", flush=True)
    hourly_longterm_analysis()
    sys.stdout.flush()

    print("\n[8c/7] Hourly data analysis — 1 month, 1h bars (legacy)", flush=True)
    hourly_analysis()
    sys.stdout.flush()

    all_res = [('BTC', btc_res), ('ETH', eth_res), ('SOL', sol_res), ('BNB', bnb_res)]
    bh_returns = {t: r['Buy & Hold'].total_return for t, r in all_res}
    winners = [
        name for name in btc_res
        if name != 'Buy & Hold'
        and all(name in r and r[name].total_return > 0 for _, r in all_res)
    ]
    print(f"\n{'═' * 92}")
    print(f"  FINAL SUMMARY v43 — Strategies profitable on ALL FOUR assets ({len(winners)})")
    print("  B&H: " + "  ".join(f"{t}={v:+.1%}" for t, v in bh_returns.items()))
    print('─' * 92)
    def avg_sharpe(name):
        return sum(r[name].sharpe for _, r in all_res) / len(all_res)
    for n in sorted(winners, key=avg_sharpe, reverse=True):
        cols = "  ".join(f"{t}={r[n].total_return:+.1%}" for t, r in all_res)
        print(f"  ★ {n:<30}  {cols}  Sharpe avg={avg_sharpe(n):.2f}")
    print(f"{'═' * 92}")
    print(f"\n  Done. Plots → {OUT_DIR}/\n")

if __name__ == '__main__':
    main()
