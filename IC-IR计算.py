"""
计算全样本 IC、IC_IR
输入：all_factors.parquet（或从清洗数据重建）
输出：ic_daily.parquet/csv, ic_summary.parquet/csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
# ============================================================
# 1. 路径配置
# ============================================================
PROJECT_ROOT = Path.cwd()
FACTOR_DIR = PROJECT_ROOT / "data" / "factors"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
FACTOR_DIR.mkdir(parents=True, exist_ok=True)

# ---- 输入：样本内 ----
FACTOR_FILE_IN = FACTOR_DIR / "all_factors_insample.parquet"

# ---- 输入：样本外 ----
FACTOR_FILE_OUT = FACTOR_DIR / "all_factors_outsample.parquet"

# ---- 输出：样本内 ----
IC_DAILY_FILE_IN = FACTOR_DIR / "ic_daily_insample.parquet"
IC_SUMMARY_FILE_IN = FACTOR_DIR / "ic_summary_insample.parquet"
IC_DAILY_CSV_IN = FACTOR_DIR / "ic_daily_insample.csv"
IC_SUMMARY_CSV_IN = FACTOR_DIR / "ic_summary_insample.csv"

# ---- 输出：样本外 ----
IC_DAILY_FILE_OUT = FACTOR_DIR / "ic_daily_outsample.parquet"
IC_SUMMARY_FILE_OUT = FACTOR_DIR / "ic_summary_outsample.parquet"
IC_DAILY_CSV_OUT = FACTOR_DIR / "ic_daily_outsample.csv"
IC_SUMMARY_CSV_OUT = FACTOR_DIR / "ic_summary_outsample.csv"

# ============================================================
##计算 IC 和 IC_IR
# ============================================================
def calculate_icir(factor_df):
    """
    计算全样本每日 IC、IC 汇总统计
    """
    df = factor_df.copy()
    
    # ---- 3.1 构建 forward_return ----
    if 'forward_return' not in df.columns:
        if 'daily_return' in df.columns:
            print("📊 从 daily_return 构建 forward_return (shift -20)...")
            df['forward_return'] = df.groupby('symbol')['daily_return'].shift(-20)
        else:
            raise KeyError("数据中缺少 daily_return，无法计算 forward_return")
    
    # 剔除 forward_return 缺失的行（最后20天）
    df = df.dropna(subset=['forward_return'])
    
    # ---- 3.2 识别因子列 ----
    exclude = ['symbol', 'date', 'daily_return', 'forward_return', 
               'open', 'high', 'low', 'close', 'vol', 'volume', 'amount', 'pct_chg']
    factor_cols = [c for c in df.columns if c not in exclude and not c.startswith('_')]
    print(f"✅ 识别到 {len(factor_cols)} 个因子列")
    
    # ---- 3.3 计算每日 Spearman IC ----
    print("📊 计算每日 IC...")
    ic_records = []
    date_groups = df.groupby('date')
    total_dates = len(date_groups)
    for idx, (dt, group) in enumerate(date_groups):
        record = {'date': dt}
        for col in factor_cols:
            # 剔除因子或 forward_return 缺失的行
            valid = group[[col, 'forward_return']].dropna()
            if len(valid) < 10:
                record[col] = np.nan
                continue
            # 跳过常数列（无方差会导致 spearmanr 报错）
            if valid[col].nunique() <= 1:
                record[col] = np.nan
                continue
            try:
                ic, _ = spearmanr(valid[col].to_numpy(), valid['forward_return'].to_numpy())
                record[col] = ic
            except Exception:
                record[col] = np.nan
        ic_records.append(record)
        if (idx+1) % 100 == 0:
            print(f"   进度：{idx+1}/{total_dates} 个交易日")
    
    ic_daily = pd.DataFrame(ic_records).set_index('date').sort_index()
    print(f"✅ 每日 IC 计算完成，共 {len(ic_daily)} 个交易日")
    
    # ---- 3.4 计算 IC 汇总统计 ----
    print("📊 计算 IC 汇总统计...")
    ic_mean = ic_daily[factor_cols].mean()
    ic_std = ic_daily[factor_cols].std()
    ic_ir_raw = (ic_mean / ic_std).fillna(0)          # 带方向，后续合成方向用
    ic_ir_abs = ic_ir_raw.abs()                       # 绝对值，筛选排名用
    ic_summary = pd.DataFrame({
        'factor': factor_cols,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir_raw,                            # ← 带符号方向
        'ic_ir_abs': ic_ir_abs,                        # ← 绝对值
        'ic_positive_ratio': (ic_daily[factor_cols] > 0).mean(),
        'ic_abs_mean': ic_daily[factor_cols].abs().mean()
    })
    ic_summary = ic_summary.sort_values('ic_ir_abs', ascending=False)
    print("✅ IC 汇总完成")

    return ic_daily, ic_summary, factor_cols


# ============================================================
# 3. 保存数据
# ============================================================
def save_icir_results(ic_daily, ic_summary,
                      ic_daily_file, ic_summary_file,
                      ic_daily_csv=None, ic_summary_csv=None):
    """
    保存 IC/ICIR 到 Parquet 和 CSV（路径全部显式传入）
    """
    # --- Parquet ---
    ic_daily.to_parquet(ic_daily_file)
    print(f"✅ 每日 IC 保存：{ic_daily_file}")

    ic_summary.to_parquet(ic_summary_file, index=False)
    print(f"✅ IC 汇总保存：{ic_summary_file}")

    # --- CSV ---
    if ic_daily_csv:
        ic_daily.to_csv(ic_daily_csv, encoding='utf-8-sig')
        print(f"✅ CSV 保存：{ic_daily_csv}")
    if ic_summary_csv:
        ic_summary.to_csv(ic_summary_csv, encoding='utf-8-sig')
        print(f"✅ CSV 保存：{ic_summary_csv}")


# ============================================================
# 4. 主程序入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📊 IC / IR 计算模块 — 样本内外分别计算")
    print("=" * 60)

    # 数据集列表：(label, factor_file, ic_daily_out, ic_summary_out, ic_daily_csv, ic_summary_csv)
    datasets = [
        ("样本内 insample", FACTOR_FILE_IN,
         IC_DAILY_FILE_IN, IC_SUMMARY_FILE_IN,
         IC_DAILY_CSV_IN, IC_SUMMARY_CSV_IN),
        ("样本外 outsample", FACTOR_FILE_OUT,
         IC_DAILY_FILE_OUT, IC_SUMMARY_FILE_OUT,
         IC_DAILY_CSV_OUT, IC_SUMMARY_CSV_OUT),
    ]

    for label, factor_file, ic_daily_out, ic_summary_out, ic_daily_csv, ic_summary_csv in datasets:
        print(f"\n{'=' * 60}")
        print(f"📊 处理 {label}")
        print(f"{'=' * 60}")

        if factor_file.exists():
            print(f"📁 加载因子数据：{factor_file}")
            factor_df = pd.read_parquet(factor_file)
            print(f"   行数：{len(factor_df):,}，列数：{len(factor_df.columns)}")
        else:
            print(f"⚠️ 因子文件不存在：{factor_file}，跳过")
            continue

        # 删除 Z-score 列
        drop_z_cols = [c for c in factor_df.columns if c.endswith('_z')]
        factor_df = factor_df.drop(columns=drop_z_cols)

        # 计算
        ic_daily, ic_summary, factor_cols = calculate_icir(factor_df)

        # 保存
        print(f"\n💾 保存 {label} 结果")
        save_icir_results(ic_daily, ic_summary,
                          ic_daily_out, ic_summary_out,
                          ic_daily_csv, ic_summary_csv)

        # 打印 Top 15
        print(f"\n📋 {label} IC/IR 汇总（Top 15）")
        print(ic_summary.head(15).to_string(index=False))

    print(f"\n{'=' * 60}")
    print("✅ 全部完成！")
    print(f"{'=' * 60}")
    print(f"  样本内 IC：{IC_DAILY_FILE_IN}")
    print(f"  样本外 IC：{IC_DAILY_FILE_OUT}")
