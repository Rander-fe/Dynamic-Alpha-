"""
方案2 -- 混合方案: 家族预算 + IC衰减动态因子排序
==================================================
与方案1(IC-IR+相关性+双层加权)比较:
  - 相同: 家族分类, 家族预算, 配额约束, get_dynamic_budget()
  - 不同: 因子选择用IC指数衰减加权(EWM)代替ICIR+相关性筛选
  - 优势: 更快适应IC变化, 无需相关性矩阵(加速), 参数更少

输出: data/factors/rolling_score_ml.parquet
"""

import pandas as pd, numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path.cwd()
FACTOR_DIR = PROJECT_ROOT / "data" / "factors"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ---- 输入：样本内 ----
FACTOR_FILE_IN = FACTOR_DIR / "all_factors_insample.parquet"
IC_DAILY_FILE_IN = FACTOR_DIR / "ic_daily_insample.parquet"
BENCHMARK_FILE_IN = RAW_DIR / "index_000300_raw_insample.parquet"

# ---- 输入：样本外 ----
FACTOR_FILE_OUT = FACTOR_DIR / "all_factors_outsample.parquet"
IC_DAILY_FILE_OUT = FACTOR_DIR / "ic_daily_outsample.parquet"
BENCHMARK_FILE_OUT = RAW_DIR / "index_000300_raw_outsample.parquet"

# ---- 输出：样本内 ----
ML_SCORE_OUTPUT_IN = FACTOR_DIR / "rolling_score_insample.parquet"
ML_LOG_OUTPUT_IN = FACTOR_DIR / "rolling_selection_log_insample.csv"

# ---- 输出：样本外 ----
ML_SCORE_OUTPUT_OUT = FACTOR_DIR / "rolling_score_outsample.parquet"
ML_LOG_OUTPUT_OUT = FACTOR_DIR / "rolling_selection_log_outsample.csv"

START_DATE = "2021-01-01"
END_DATE = "2026-06-30"
IC_HALFLIFE = 42       # IC指数衰减半衰期
LOOKBACK_DAYS = 252     # IC回溯期
MIN_ABS_IC = 0.015      # |IC|筛选阈值

EXCLUDE_COLS = [
    'ts_code', 'date', 'open', 'high', 'low', 'close', 'pre_close',
    'change', 'pct_chg', 'vol', 'amount', 'amount_billion',
    'name', 'daily_return', 'symbol', 'year_month',
    'float_share',
]

# 因子家族分类（与创建因子.py 完全对齐，共7家族×31因子）
FACTOR_FAMILIES = {
    'momentum': ['momentum_'],
    'quality': ['roe', 'grossprofit_margin', 'netprofit_yoy', 'revenue_yoy', 'debt_to_assets'],
    'valuation': ['pe_ttm', 'pb', 'ps_ttm', 'total_mv'],
    'liquidity': ['turnover_rate', 'avg_turnover_'],
    'volatility': ['volatility_'],
    'drawdown': ['max_drawdown_'],
    'technical': ['volume_ratio', 'volume_change'],
}

# 配额约束
QUOTA = {
    'momentum': {'min': 2, 'max': 4},
    'drawdown': {'min': 0, 'max': 2},
    'volatility': {'min': 0, 'max': 2},
    'quality': {'min': 1, 'max': 5},
    'valuation': {'min': 0, 'max': 3},
    'liquidity': {'min': 0, 'max': 3},
    'technical': {'min': 0, 'max': 2},
}




def get_factor_family(name):
    for fam, kws in FACTOR_FAMILIES.items():
        if any(kw in name for kw in kws):
            return fam
    return 'technical'



def compute_family_icir(ic_daily, factor_families, current_date=None, lookback=252):
    """
    计算各因子家族的 IC_IR（信息比率），并归一化为权重。

    Parameters:
        ic_daily: DataFrame，索引为日期，列为原始因子名，值为日频 IC
        factor_families: 因子家族字典
        current_date: 当前日期，若提供则剔除最近 IC_FORWARD_DAYS 避免未来函数
        lookback: 回溯天数（默认 252）

    Returns:
        dict: {家族名: 权重}，权重和为 1
    """
    # 若指定 current_date，剔除最近 IC_FORWARD_DAYS 避免未来函数
    if current_date is not None:
        recent_cutoff = current_date - pd.offsets.BDay(IC_FORWARD_DAYS)
        ic = ic_daily[ic_daily.index <= recent_cutoff]
    else:
        ic = ic_daily

    family_icir = {}
    for family, keywords in factor_families.items():
        cols = [c for c in ic.columns if any(kw in c for kw in keywords)]
        if not cols:
            family_icir[family] = 0.0
            continue

        # 家族日频 IC = 内部因子 IC 的等权平均
        family_ic = ic[cols].mean(axis=1, skipna=True)
        hist = family_ic.dropna().tail(lookback)
        if len(hist) < 60:
            family_icir[family] = 0.0
            continue

        mean_ic = hist.mean()
        std_ic = hist.std()
        if std_ic == 0:
            family_icir[family] = 0.0
        else:
            family_icir[family] = abs(mean_ic / std_ic)

    total = sum(family_icir.values())
    if total == 0:
        n = len(factor_families)
        return {fam: 1.0 / n for fam in factor_families}
    else:
        return {fam: val / total for fam, val in family_icir.items()}


def get_market_regime(current_date, bench_series):
    """
    基于 MA60 与 MA120 判断当前市场状态。
    返回: 'bull'（强势）, 'bear'（弱势）, 'neutral'（震荡）
    """
    hist = bench_series[bench_series.index <= current_date].tail(240)
    if len(hist) < 120:
        return 'neutral'

    ma60 = hist.tail(60).mean()
    ma120 = hist.tail(120).mean()
    price = hist.iloc[-1]

    if ma60 > ma120 and price > ma60:
        return 'bull'
    elif ma60 < ma120 and price < ma60:
        return 'bear'
    else:
        return 'neutral'


def _aggregate_family_ic(ic_daily, factor_families, lookback=252):
    """将因子级IC聚合为家族级日频IC序列"""
    family_ic = {}
    for family, keywords in factor_families.items():
        cols = [c for c in ic_daily.columns if any(kw in c for kw in keywords)]
        if cols:
            family_ic[family] = ic_daily[cols].mean(axis=1, skipna=True)
    df = pd.DataFrame(family_ic).tail(lookback).dropna()
    return df


def _compute_ewma_covariance(values, halflife=63):
    """计算EWMA协方差矩阵"""
    lam = np.exp(np.log(0.5) / halflife)
    n = len(values)
    w = lam ** np.arange(n - 1, -1, -1)
    w = w / w.sum()

    demeaned = values - values.mean(axis=0)
    cov = np.dot((w[:, np.newaxis] * demeaned).T, demeaned) / (1 - (w**2).sum())
    return cov


def _compute_gmv_weights(cov_matrix):
    """GMV权重（全局最小方差）—— 使用伪逆增强数值稳定性"""
    inv = np.linalg.pinv(cov_matrix.values)
    ones = np.ones(len(cov_matrix))
    w = inv @ ones / (ones @ inv @ ones)
    return pd.Series(w, index=cov_matrix.index)


def get_risk_budget_weights(ic_daily, factor_families, current_date=None,
                             method='gmv', lookback=252, halflife=63):
    """
    计算风险平价/GMV权重（家族级别）。

    Parameters:
        ic_daily: DataFrame，索引为日期，列为原始因子名，值为日频IC
        factor_families: 因子家族字典
        current_date: 当前日期，若提供则剔除最近IC_FORWARD_DAYS避免未来函数
        method: 'gmv'（全局最小方差）
        lookback: 回溯天数
        halflife: EWMA半衰期

    Returns:
        dict: {家族名: 权重}，权重和为1
    """
    if current_date is not None:
        recent_cutoff = current_date - pd.offsets.BDay(IC_FORWARD_DAYS)
        ic = ic_daily[ic_daily.index <= recent_cutoff]
    else:
        ic = ic_daily

    family_ic_df = _aggregate_family_ic(ic, factor_families, lookback)
    if len(family_ic_df) < 60:
        n = len(factor_families)
        return {fam: 1.0 / n for fam in factor_families}

    cov = _compute_ewma_covariance(family_ic_df.values, halflife)
    cov_df = pd.DataFrame(cov, index=family_ic_df.columns, columns=family_ic_df.columns)

    w = _compute_gmv_weights(cov_df)
    w = w.clip(0)          # 截断负权重
    w = w / w.sum()        # 归一化
    return w.to_dict()


# IC基于shift(-20)前向收益，最近20个交易日的IC含有未来信息，需剔除
IC_FORWARD_DAYS = 20


def compute_ewm_ic_weights(ic_daily, current_date, factor_list):
    """
    对每个因子计算过去LOOKBACK_DAYS的IC指数衰减加权平均
    返回: {factor: smoothed_ic}

    注意: 因IC使用shift(-20)前向收益计算，故剔除最近IC_FORWARD_DAYS天数据，
          避免前向收益跨越current_date造成未来函数。
    """
    # 最远回溯边界（交易日，与LOOKBACK_DAYS=252交易日保持一致）
    cutoff = current_date - pd.offsets.BDay(LOOKBACK_DAYS)
    # 最近剔除边界: 最后IC_FORWARD_DAYS个交易日(前向收益窗口)不可用
    recent_cutoff = current_date - pd.offsets.BDay(IC_FORWARD_DAYS)
    hist = ic_daily[(ic_daily.index <= recent_cutoff) & (ic_daily.index > cutoff)]

    result = {}
    for fc in factor_list:
        if fc not in hist.columns:
            continue
        ics = hist[fc].dropna().values
        if len(ics) < 10:
            continue
        n = len(ics)
        decay = np.exp(np.log(0.5) * np.arange(n - 1, -1, -1) / IC_HALFLIFE)
        decay = decay / decay.sum()
        result[fc] = np.sum(ics * decay)

    return result


# 需要兜底配额的核心家族
CORE_FAMILIES = {'momentum', 'quality'}

QUOTA_FALLBACK_MIN_ABS_IC = 0.005  # 兜底放宽至 |IC| > 0.005，避免硬塞纯噪音


def select_factors_per_family(ic_weights, family):
    """在指定家族内选最优因子, 满足配额约束（核心家族有兜底）"""
    family_factors = {f: ic_weights[f] for f in ic_weights if get_factor_family(f) == family}
    if not family_factors:
        return []

    quota = QUOTA.get(family, {'min': 0, 'max': 3})
    # 按|IC|降序
    sorted_f = sorted(family_factors.items(), key=lambda x: abs(x[1]), reverse=True)
    # 过滤|IC| >= MIN_ABS_IC
    valid = [(f, w) for f, w in sorted_f if abs(w) >= MIN_ABS_IC]

    # 核心家族兜底: 若 valid 不足 min，从剩余因子中放宽阈值补足
    if family in CORE_FAMILIES and len(valid) < quota['min']:
        remaining = [(f, w) for f, w in sorted_f
                     if abs(w) < MIN_ABS_IC and abs(w) >= QUOTA_FALLBACK_MIN_ABS_IC]
        shortage = quota['min'] - len(valid)
        valid = valid + remaining[:shortage]

    # 至多选quota['max']个
    selected = valid[:quota['max']]

    return selected  # [(factor, ic_weight), ...]


def load_data(label, factor_file, ic_file, bench_file):
    """加载指定数据集的因子/IC/基准"""
    print("=" * 60)
    print(f"[方案2-Hybrid] 加载数据 — {label}")
    print("=" * 60)

    factor_df = pd.read_parquet(factor_file)
    factor_df['date'] = pd.to_datetime(factor_df['date'])

    ic_daily = pd.read_parquet(ic_file)

    bench = pd.read_parquet(bench_file)
    bench['date'] = pd.to_datetime(bench['date'])
    bench_series = bench.set_index('date')['index_close']

    print(f"   因子: {len(factor_df):,}行 ({factor_df['date'].min().date()} ~ {factor_df['date'].max().date()})")
    print(f"   IC日频: {len(ic_daily)}天, 基准: {len(bench_series)}天")
    return factor_df, ic_daily, bench_series


def get_factor_columns(factor_df):
    cols = [c for c in factor_df.columns if c not in EXCLUDE_COLS]
    return [c for c in cols if pd.api.types.is_numeric_dtype(factor_df[c])]


def hybrid_scoring(factor_df, ic_daily, bench_series, label="",
                   score_output=None, log_output=None):
    """
    家族预算 + IC-EWM 因子选择与评分。

    Parameters:
        label:        数据集标签（如 "样本内insample"）
        score_output: 评分输出路径（parquet）
        log_output:   日志输出路径（csv）
    """
    if score_output is None:
        score_output = ML_SCORE_OUTPUT_IN
    if log_output is None:
        log_output = ML_LOG_OUTPUT_IN

    print(f"\n[方案2-Hybrid] 家族预算+IC-EWM因子选择 — {label}...")

    factor_cols = get_factor_columns(factor_df)
    print(f"   因子总数：{len(factor_cols)}")

    df = factor_df.copy()
    df['year_month'] = df['date'].dt.to_period('M')
    all_months = sorted(df['year_month'].unique())
    total_months = len(all_months)

    scored_list = []
    log_records = []

    for month_idx, ym in enumerate(all_months):
        month_data = df[df['year_month'] == ym]
        last_date = month_data['date'].max()

        # 获取月末因子值(最后交易日)
        month_end = month_data[month_data['date'] == last_date].copy()
        if len(month_end) == 0:
            continue

        # 1. 计算IC-EWM权重
        ic_weights = compute_ewm_ic_weights(ic_daily, last_date, factor_cols)

        if len(ic_weights) == 0:
            log_records.append({
                'year_month': ym, 'n_factors': 0,
                'factors': '', 'fallback': True, 'zero_factor': True,
                'model': 'Hybrid', 'budget': 'N/A'
            })
            continue

        # 2. 按家族选因子
        selected_factors = {}  # {factor: ic_weight}
        for family in FACTOR_FAMILIES:
            picks = select_factors_per_family(ic_weights, family)
            for f, w in picks:
                selected_factors[f] = w

        # ========== 新预算逻辑：ICIR + 风险平价混合 ==========

        # 3a. 计算 ICIR 权重（家族级别）
        icir_weights = compute_family_icir(ic_daily, FACTOR_FAMILIES,
                                           current_date=last_date, lookback=252)

        # 3b. 计算风险平价权重（家族级别，GMV方法）
        rp_weights = get_risk_budget_weights(ic_daily, FACTOR_FAMILIES,
                                              current_date=last_date, method='gmv')

        # 3c. 判断市场状态（用于决定混合比例）
        regime = get_market_regime(last_date, bench_series)

        if regime == 'bull':
            alpha = 0.95      # 牛市：近乎纯 ICIR，GMV 拖累降到最低
        elif regime == 'bear':
            alpha = 0.25      # 熊市：重风险平价（强防御）
        else:
            alpha = 0.55      # 震荡：略偏 ICIR

        # 3d. 混合权重
        all_fams = set(icir_weights.keys()) | set(rp_weights.keys())
        mixed = {}
        for fam in all_fams:
            icir_w = icir_weights.get(fam, 0.0)
            rp_w = rp_weights.get(fam, 0.0)
            mixed[fam] = alpha * icir_w + (1 - alpha) * rp_w

        # 归一化
        total_mix = sum(mixed.values())
        if total_mix > 0:
            budget = {k: v / total_mix for k, v in mixed.items()}
        else:
            n = len(all_fams)
            budget = {fam: 1.0 / n for fam in all_fams}

        # 4. 按家族分组, 分配家族内权重
        # 因子权重 = 家族预算 * (|因子IC| / 家族内|IC|之和)
        family_groups = {}
        for fc, ic_w in selected_factors.items():
            fam = get_factor_family(fc)
            if fam not in family_groups:
                family_groups[fam] = []
            family_groups[fam].append((fc, ic_w))

        # 5. 截面Z-score
        features = month_end[factor_cols].copy()
        f_mean = features.mean()
        f_std = features.std().replace(0, 1.0)
        features_z = (features - f_mean) / f_std
        features_z = features_z.fillna(0).clip(-4, 4)

        # 6. 合成评分
        composite = pd.Series(0.0, index=month_end.index)
        active_factors = []

        for fam, fam_factors in family_groups.items():
            fam_budget = budget.get(fam, 0.0)
            if fam_budget <= 0 or len(fam_factors) == 0:
                continue

            # 家族内权重: |IC|归一化
            total_abs_ic = sum(abs(w) for _, w in fam_factors)
            if total_abs_ic == 0:
                continue

            for fc, ic_w in fam_factors:
                # 最终权重 = 家族预算 * (|IC|/家族内|IC|之和)
                weight = fam_budget * (abs(ic_w) / total_abs_ic)
                # 方向由IC符号决定
                direction = np.sign(ic_w)
                if fc in features_z.columns:
                    composite += features_z[fc].values * direction * weight
                    active_factors.append(fc)

        month_end['composite_score'] = composite.values
        scored_list.append(month_end[['date', 'symbol', 'year_month', 'composite_score']])

        log_records.append({
            'year_month': ym, 'n_factors': len(active_factors),
            'factors': ','.join(active_factors[:8]),
            'fallback': False, 'zero_factor': len(active_factors) == 0,
            'model': 'Hybrid',
            'budget': f'alpha={alpha:.2f}',
            'top_ic_factor': max(selected_factors, key=lambda x: abs(selected_factors[x])) if selected_factors else ''
        })

        if (month_idx + 1) % 12 == 0 or month_idx == total_months - 1:
            n_active = len(active_factors)
            fam_info = {fam: len([f for f in active_factors if get_factor_family(f) == fam]) for fam in FACTOR_FAMILIES}
            print(f"   [{month_idx+1}/{total_months}] {str(ym)} | {n_active}因子 | alpha={alpha:.2f} | {fam_info}")

    scored_df = pd.concat(scored_list, ignore_index=True)
    scored_df['date'] = pd.to_datetime(scored_df['date'])

    print(f"\n[方案2-Hybrid] 完成")
    print(f"   评分行数: {len(scored_df):,}, 覆盖月数: {scored_df['year_month'].nunique()}")
    print(f"   日期范围: {scored_df['date'].min().date()} ~ {scored_df['date'].max().date()}")

    scored_df.to_parquet(score_output, index=False)
    print(f"   已保存: {score_output}")

    log_df = pd.DataFrame(log_records)
    log_df.to_csv(log_output, index=False, encoding='utf-8-sig')
    print(f"   已保存: {log_output}")

    return scored_df, log_df


if __name__ == "__main__":
    print("=" * 60)
    print("方案2 -- 混合方案(家族预算+IC-EWM)")
    print(f"   IC半衰期: {IC_HALFLIFE}天, 最小|IC|: {MIN_ABS_IC}")
    print("=" * 60)

    # ---- 样本内 ----
    factor_df_in, ic_daily_in, bench_in = load_data(
        "样本内 insample", FACTOR_FILE_IN, IC_DAILY_FILE_IN, BENCHMARK_FILE_IN)
    scored_in, log_in = hybrid_scoring(
        factor_df_in, ic_daily_in, bench_in, "样本内 insample",
        ML_SCORE_OUTPUT_IN, ML_LOG_OUTPUT_IN)

    # ---- 样本外 ----
    factor_df_out, ic_daily_out, bench_out = load_data(
        "样本外 outsample", FACTOR_FILE_OUT, IC_DAILY_FILE_OUT, BENCHMARK_FILE_OUT)
    scored_out, log_out = hybrid_scoring(
        factor_df_out, ic_daily_out, bench_out, "样本外 outsample",
        ML_SCORE_OUTPUT_OUT, ML_LOG_OUTPUT_OUT)

    print(f"\n{'=' * 60}")
    print(f"[OK] 方案2-Hybrid 完成")
    print(f"{'=' * 60}")
    print(f"  样本内评分：{ML_SCORE_OUTPUT_IN}")
    print(f"  样本外评分：{ML_SCORE_OUTPUT_OUT}")
