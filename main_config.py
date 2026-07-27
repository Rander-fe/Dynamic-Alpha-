# ============================================================
# 📋 主配置 —— 集中管理所有路径、参数、Token
# ============================================================
# 所有模块统一从此文件读取配置，避免路径/参数不一致
# ============================================================

import os
import sys
from pathlib import Path

# ========================
# 1. 项目根目录 & 数据目录
# ========================
# 自动检测项目根目录（支持从子模块导入）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
CLEAN_DIR      = DATA_DIR / "clean"
FACTOR_DIR     = DATA_DIR / "factors"
ICIR_DIR       = DATA_DIR / "icir_selection"
SELECTION_DIR  = DATA_DIR / "selection"
CONFIG_DIR     = PROJECT_ROOT / "config"

# 确保目录存在
for d in [RAW_DIR, CLEAN_DIR, FACTOR_DIR, ICIR_DIR, SELECTION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========================
# 2. Tushare Token
# ========================
TUSHARE_TOKEN = os.environ.get(
    "TUSHARE_TOKEN",
    "ba3ef3862d622efeb8a0214b8ea6dfac00809e234af80b2c40035e8c"
)

# ========================
# 3. 时间范围
# ========================
START_DATE = "2021-01-01"
END_DATE   = "2026-06-30"

# ========================
# 4. 因子计算参数
# ========================
TECHNICAL_WINDOWS = [5, 10, 20, 60, 120, 240]
WINSORIZE_QUANTILE = 0.99
FILLNA_METHOD = 'mean'
APPLY_STANDARDIZE = True

# ========================
# 5. 动态选因子参数
# ========================
IC_LOOKBACK_DAYS = 252        # 滚动 IC 窗口（≈12个交易日）
MIN_ABS_IC = 0.02             # |IC| 初筛阈值
CORR_THRESHOLD = 0.6          # 相关性贪心筛选阈值
TOP_N = 30                    # 每月选股数
TURNOVER_BUFFER = 10          # 换手缓冲

# ========================
# 6. 回测参数
# ========================
INITIAL_CAPITAL = 1_000_000   # 初始资金
COST_RATE = 0.0003            # 交易成本（万分之三）

# ========================
# 7. 文件命名
# ========================
DAILY_RAW_FILE     = "stock_daily_raw"
BASIC_RAW_FILE     = "stock_basic_raw"
FINA_RAW_FILE      = "stock_fina_raw"

DAILY_CLEAN_FILE   = "stock_daily_clean"
BASIC_CLEAN_FILE   = "stock_basic_clean"
FINA_CLEAN_FILE    = "stock_fina_clean"

FACTOR_FILE        = "all_factors"

ICIR_SELECTION_FILE    = "icir_selection_log"
ICIR_SCORED_FILE       = "icir_scored"
ICIR_SELECTED_FACTORS  = "selected_factors_per_month"

NAV_FILE           = SELECTION_DIR / "backtest_nav.parquet"
PERFORMANCE_FILE   = SELECTION_DIR / "performance_summary.csv"


def print_config():
    """打印当前配置摘要"""
    lines = [
        "=" * 60,
        "  CONFIG SUMMARY",
        "=" * 60,
        f"Project Root:   {PROJECT_ROOT}",
        f"Raw Data:       {RAW_DIR}",
        f"Clean Data:     {CLEAN_DIR}",
        f"Factor Data:    {FACTOR_DIR}",
        f"ICIR Selection: {ICIR_DIR}",
        f"Date Range:     {START_DATE} ~ {END_DATE}",
        f"IC Window(days):{IC_LOOKBACK_DAYS}",
        f"|IC| Threshold: {MIN_ABS_IC}",
        f"Corr Threshold: {CORR_THRESHOLD}",
        f"Top N + Buffer: {TOP_N} + {TURNOVER_BUFFER}",
        "=" * 60,
    ]
    for line in lines:
        print(line)


def main():
    """主函数：打印配置摘要 + 回测绩效报告 + 分年收益"""
    import pandas as pd

    # 同时写入日志文件，避免终端编码问题
    log_path = PROJECT_ROOT / "output.log"
    with open(log_path, 'w', encoding='utf-8') as log:

        def tee(msg):
            print(msg)
            log.write(msg + '\n')

        tee("=" * 60)
        tee("  CONFIG + PERFORMANCE REPORT")
        tee("=" * 60)
        tee(f"Project Root: {PROJECT_ROOT}")
        tee(f"Date Range:   {START_DATE} ~ {END_DATE}")
        tee("")

        print_config()
        # 也把配置信息写入日志
        log.write("=" * 60 + '\n')
        log.write("  CONFIG SUMMARY\n")
        log.write(f"Project Root:   {PROJECT_ROOT}\n")
        log.write(f"Raw Data:       {RAW_DIR}\n")
        log.write(f"Clean Data:     {CLEAN_DIR}\n")
        log.write(f"Factor Data:    {FACTOR_DIR}\n")
        log.write(f"ICIR Selection: {ICIR_DIR}\n")
        log.write(f"Date Range:     {START_DATE} ~ {END_DATE}\n")
        log.write(f"IC Window(days):{IC_LOOKBACK_DAYS}\n")
        log.write(f"|IC| Threshold: {MIN_ABS_IC}\n")
        log.write(f"Corr Threshold: {CORR_THRESHOLD}\n")
        log.write(f"Top N + Buffer: {TOP_N} + {TURNOVER_BUFFER}\n")
        log.write("=" * 60 + '\n')

        # ----- 绩效报告 -----
        tee("")
        tee("=" * 60)
        tee("  BACKTEST PERFORMANCE")
        tee("=" * 60)

        if PERFORMANCE_FILE.exists():
            perf = pd.read_csv(PERFORMANCE_FILE).iloc[0].to_dict()
            tee(f"Cumulative Return:      {perf['total_return']*100:.2f}%")
            tee(f"Annualized Return:      {perf['annual_return']*100:.2f}%")
            tee(f"Annualized Volatility:  {perf['annual_volatility']*100:.2f}%")
            tee(f"Sharpe Ratio:           {perf['sharpe_ratio']:.4f}")
            tee(f"Max Drawdown:           {perf['max_drawdown']*100:.2f}%")
            tee(f"Win Rate:               {perf['win_rate']*100:.2f}%")
            if perf.get('excess_return') is not None and not pd.isna(perf['excess_return']):
                tee(f"Excess Return(vs CSI300): {perf['excess_return']*100:.2f}%")
            if perf.get('information_ratio') is not None and not pd.isna(perf['information_ratio']):
                tee(f"Information Ratio:      {perf['information_ratio']:.4f}")
        else:
            tee(f"[WARN] Performance file not found: {PERFORMANCE_FILE}")
            tee("       Run 回测.py first to generate it.")

        # ----- 分年收益 -----
        if NAV_FILE.exists():
            nav_df = pd.read_parquet(NAV_FILE)
            nav_df['date'] = pd.to_datetime(nav_df['date'])
            nav_df['year'] = nav_df['date'].dt.year
            tee("")
            tee("-" * 70)
            tee("  YEARLY RETURNS")
            tee("-" * 70)
            yearly = nav_df.groupby('year').agg(
                start_nav=('nav', 'first'),
                end_nav=('nav', 'last'),
                max_dd=('nav', lambda x: ((x / x.cummax()) - 1).min()),
            )
            yearly['Strategy_Ret'] = yearly['end_nav'] / yearly['start_nav'] - 1
            yearly['Cum_Nav'] = (1 + yearly['Strategy_Ret']).cumprod()
            header = f"{'Year':<8}{'Return':>10}{'Cum_Nav':>14}{'Max_DD':>12}"
            tee(header)
            for yr, row in yearly.iterrows():
                tee(f"{yr:<8}{row['Strategy_Ret']*100:>9.2f}%{row['Cum_Nav']:>13.4f}{row['max_dd']*100:>11.2f}%")
        else:
            tee(f"[WARN] NAV file not found: {NAV_FILE}")

        tee("")
        tee("=" * 60)
        tee(f"[OK] Log also saved to: {log_path}")
        tee("=" * 60)

    print("")


if __name__ == "__main__":
    main()
