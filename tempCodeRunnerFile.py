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
CONFIG_DIR     = PROJECT_ROOT / "config"

# 确保目录存在
for d in [RAW_DIR, CLEAN_DIR, FACTOR_DIR, ICIR_DIR]:
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


def print_config():
    """打印当前配置摘要"""
    print("=" * 60)
    print("📋 配置摘要")
    print("=" * 60)
    print(f"项目根目录:    {PROJECT_ROOT}")
    print(f"原始数据:      {RAW_DIR}")
    print(f"清洗数据:      {CLEAN_DIR}")
    print(f"因子数据:      {FACTOR_DIR}")
    print(f"ICIR 选因子:   {ICIR_DIR}")
    print(f"时间范围:      {START_DATE} ~ {END_DATE}")
    print(f"IC 窗口(天):   {IC_LOOKBACK_DAYS}")
    print(f"|IC| 阈值:     {MIN_ABS_IC}")
    print(f"相关性阈值:    {CORR_THRESHOLD}")
    print(f"每月选股:      {TOP_N} + 缓冲 {TURNOVER_BUFFER}")
    print("=" * 60)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    print_config()
    sys.stdout.flush()
