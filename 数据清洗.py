# ============================================================
# 🧹 数据清洗 —— 批量清洗沪深300股票数据
# ============================================================
# 功能：
#   1. 导入依赖与路径配置
#   2. 日线数据清洗：去重、去空、过滤、计算收益率（重命名：clean_daily_market_data）
#   3. 估值数据清洗：标准化、去极值
#   4. 财务数据清洗：类型转换、去重
#   5. 数据对齐与合并：merge_asof 前向填充财务数据（merge_and_align_data）
#   6. 数据概览报告
#   7. 保存清洗后数据
# ============================================================
# 时间范围：2018-01-01 ~ 2026-06-30
# ============================================================

import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ========================
# 1. 环境与路径配置
# ========================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

# 确保目录存在
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ---- 输入文件（原始数据） ----
INPUT_DAILY  = RAW_DIR / "stock_daily_raw.csv"
INPUT_BASIC  = RAW_DIR / "stock_basic_raw.csv"
INPUT_FINA   = RAW_DIR / "stock_fina_raw.csv"
INPUT_INDEX  = RAW_DIR / "index_000300_raw.csv"

# ---- 输出文件（清洗后全量） ----
OUTPUT_DAILY  = CLEAN_DIR / "stock_daily_clean.csv"
OUTPUT_BASIC  = CLEAN_DIR / "stock_basic_clean.csv"
OUTPUT_FINA   = CLEAN_DIR / "stock_fina_clean.csv"
OUTPUT_MERGED = CLEAN_DIR / "stock_merged_clean.csv"

# ---- 输出文件（样本内 insample） ----
OUTPUT_DAILY_IN  = CLEAN_DIR / "stock_daily_clean_insample.csv"
OUTPUT_BASIC_IN  = CLEAN_DIR / "stock_basic_clean_insample.csv"
OUTPUT_FINA_IN   = CLEAN_DIR / "stock_fina_clean_insample.csv"
OUTPUT_MERGED_IN = CLEAN_DIR / "stock_merged_clean_insample.csv"

# ---- 输出文件（样本外 outsample） ----
OUTPUT_DAILY_OUT  = CLEAN_DIR / "stock_daily_clean_outsample.csv"
OUTPUT_BASIC_OUT  = CLEAN_DIR / "stock_basic_clean_outsample.csv"
OUTPUT_FINA_OUT   = CLEAN_DIR / "stock_fina_clean_outsample.csv"
OUTPUT_MERGED_OUT = CLEAN_DIR / "stock_merged_clean_outsample.csv"

# ---- 指数拆分输出 ----
OUTPUT_INDEX_IN  = RAW_DIR / "index_000300_raw_insample.csv"
OUTPUT_INDEX_OUT = RAW_DIR / "index_000300_raw_outsample.csv"

print(f"📁 项目根目录：{PROJECT_ROOT}")
print(f"📁 原始数据目录：{RAW_DIR}")
print(f"📁 清洗数据目录：{CLEAN_DIR}")
print(f"📥 输入文件：{INPUT_DAILY.name}, {INPUT_BASIC.name}, {INPUT_FINA.name}, {INPUT_INDEX.name}")
print(f"📤 输出文件(全量)：{OUTPUT_DAILY.name}, {OUTPUT_BASIC.name}, {OUTPUT_FINA.name}, {OUTPUT_MERGED.name}")
print(f"📤 输出文件(样本内)：{OUTPUT_DAILY_IN.name}, {OUTPUT_BASIC_IN.name}, {OUTPUT_FINA_IN.name}, {OUTPUT_MERGED_IN.name}")
print(f"📤 输出文件(样本外)：{OUTPUT_DAILY_OUT.name}, {OUTPUT_BASIC_OUT.name}, {OUTPUT_FINA_OUT.name}, {OUTPUT_MERGED_OUT.name}")

# ========================
# 2. 时间范围
# ========================
START_DATE = "2018-01-01"
END_DATE   = "2026-06-30"

print(f"📅 数据时间范围：{START_DATE} ~ {END_DATE}")

# ========================
# 3. 清洗函数定义
# ========================

def clean_daily_market_data(df: pd.DataFrame, min_trade_days: int = 60) -> pd.DataFrame:
    """
    清洗股票日线数据：去重、去空、转换类型、剔除交易天数不足的股票、计算收益率。

    重命名说明：原函数名 clean_stock_data → 更名 clean_daily_market_data，
    明确表示清洗的是"日线市场数据"而非其他类型数据。

    Parameters:
        df: 原始日线数据 DataFrame（需含 symbol, date, close 等列）
        min_trade_days: 最小交易天数阈值，默认60日（用于过滤次新股/ST股）

    Returns:
        清洗后的 DataFrame，新增 daily_return 和 amount_billion 列
    """
    df = df.copy()

    # 处理 symbol 列
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        df = df.dropna(subset=['symbol'])
    else:
        raise KeyError("日线数据缺少 'symbol' 列")

    # 删除关键字段缺失的行
    df = df.dropna(subset=["date", "symbol", "close"])

    # 删除重复记录（同一天同一只股票）
    df = df.drop_duplicates(subset=["date", "symbol"])

    # 数值列转换
    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pct_chg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除关键数值列为 NaN 的行
    key_numeric_cols = ["open", "close", "volume", "amount", "pct_chg"]
    existing_key_cols = [c for c in key_numeric_cols if c in df.columns]
    df = df.dropna(subset=existing_key_cols)

    # 按股票和日期排序
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 剔除数据集中交易天数不足 min_trade_days 的股票（自动过滤次新股与 ST 股）
    df["_trade_days"] = df.groupby("symbol").cumcount() + 1
    valid_symbols = df.groupby("symbol")["_trade_days"].max()
    valid_symbols = valid_symbols[valid_symbols >= min_trade_days].index
    before = df["symbol"].nunique()
    df = df[df["symbol"].isin(valid_symbols)]
    after = df["symbol"].nunique()
    df = df.drop(columns=["_trade_days"])
    print(f"  交易天数过滤：剔除 {before - after} 只股票（<{min_trade_days}日），保留 {after} 只")

    # 计算每日收益率
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    # 计算成交额（亿元）—— Tushare amount 单位为「千元」，除以 1e5 换算为亿元
    if "amount" in df.columns:
        df["amount_billion"] = df["amount"] / 1e5

    # 去除收益率缺失的行（每只股票的第一天）
    df = df.dropna(subset=["daily_return"])

    return df


def clean_valuation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗每日估值数据。

    Parameters:
        df: 原始估值数据 DataFrame

    Returns:
        清洗后的 DataFrame
    """
    df = df.copy()

    # 处理 symbol 列
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        df = df.dropna(subset=['symbol'])
    else:
        raise KeyError("估值数据缺少 'symbol' 列")

    df = df.dropna(subset=["date", "symbol"])
    df = df.drop_duplicates(subset=["date", "symbol"])

    # 数值列转换
    numeric_cols = ["total_mv", "float_share", "pe_ttm", "pb", "ps_ttm", "turnover_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除 PE/PB/市值为空的行
    key_cols = ["pe_ttm", "pb", "total_mv"]
    existing = [c for c in key_cols if c in df.columns]
    df = df.dropna(subset=existing)

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗季度财务数据。

    Parameters:
        df: 原始财务数据 DataFrame

    Returns:
        清洗后的 DataFrame
    """
    df = df.copy()

    # 处理 symbol 列
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).str.zfill(6)
        df = df.dropna(subset=['symbol'])
    else:
        raise KeyError("财务数据缺少 'symbol' 列")

    df = df.dropna(subset=["date", "symbol"])
    df = df.drop_duplicates(subset=["date", "symbol"])

    # 数值列转换
    numeric_cols = [
        "roe", "netprofit_yoy", "revenue_yoy", "grossprofit_margin",
        "debt_to_assets", "net_assets", "net_profit", "revenue"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


# ========================
# 4. 数据对齐与合并（merge_asof 前向填充财务数据）
# ========================

def merge_and_align_data(daily: pd.DataFrame, basic: pd.DataFrame,
                         fina: pd.DataFrame) -> pd.DataFrame:
    """
    将日线、估值、财务数据合并为统一宽表。

    合并策略：
      - 日线 + 估值：精确匹配 symbol + date（left join）
      - + 财务数据：merge_asof by symbol，direction='backward'
        按报告期前向填充到每个交易日，严格避免未来函数

    Parameters:
        daily: 清洗后的日线数据
        basic: 清洗后的估值数据
        fina:  清洗后的财务数据（季度频次）

    Returns:
        合并后的宽表 DataFrame
    """
    print("\n" + "=" * 60)
    print("🔗 数据对齐与合并（merge_asof）")
    print("=" * 60)

    # Step 1: 日线 + 估值（精确匹配日期）
    print("\n📌 Step 1: 日线 + 估值 merge(on=['symbol','date'], how='left')")
    merged = daily.merge(
        basic[['symbol', 'date', 'total_mv', 'float_share', 'pe_ttm', 'pb', 'ps_ttm']],
        on=['symbol', 'date'], how='left'
    )
    print(f"   合并后：{len(merged):,} 行 × {len(merged.columns)} 列")

    # Step 2: + 财务数据（merge_asof 前向填充）
    print("\n📌 Step 2: + 财务数据 merge_asof(direction='backward')")
    fina_cols = ['symbol', 'date'] + [c for c in
        ['roe', 'netprofit_yoy', 'revenue_yoy', 'grossprofit_margin', 'debt_to_assets']
        if c in fina.columns]

    # 按 symbol 分组逐组 merge_asof，避免 object 类型列排序检测不可靠的问题
    merged = merged.sort_values(['symbol', 'date']).reset_index(drop=True)
    fina_sorted = fina[fina_cols].sort_values(['symbol', 'date']).reset_index(drop=True)

    def _asof_merge_group(grp: pd.DataFrame) -> pd.DataFrame:
        """对单只股票的日线数据，按报告期前向填充财务数据"""
        # groupby 排除了 symbol 列，用 grp.name 获取分组键
        sym = grp.name
        fina_grp = fina_sorted[fina_sorted['symbol'] == sym]
        # 排除 symbol 列，避免 merge_asof 产生 _x/_y 后缀或丢列
        fina_grp_clean = fina_grp.drop(columns=['symbol']).sort_values('date')
        result = pd.merge_asof(
            grp.sort_values('date'),
            fina_grp_clean,
            on='date',
            direction='backward'
        ) if not fina_grp.empty else grp
        # 补回 groupby 排除掉的 symbol 列，否则下游因子计算找不到 symbol
        result['symbol'] = sym
        return result

    merged = merged.groupby('symbol', group_keys=False).apply(
        _asof_merge_group
    ).reset_index(drop=True)

    # 统计财务因子覆盖率
    fina_factor_cols = [c for c in ['roe', 'netprofit_yoy', 'revenue_yoy',
                                     'grossprofit_margin', 'debt_to_assets']
                        if c in merged.columns]
    for col in fina_factor_cols:
        coverage = merged[col].notna().mean()
        print(f"   {col} 覆盖率：{coverage:.1%}")

    print(f"\n✅ 数据对齐完成：{len(merged):,} 行 × {len(merged.columns)} 列")
    return merged


# ========================
# 5. 数据清洗主函数
# ========================

def run_data_cleaning(raw_dir=None, clean_dir=None, save=True):
    """
    主清洗流程：加载原始数据 → 清洗 → 保存。

    Parameters:
        raw_dir: 原始数据目录，默认 RAW_DIR
        clean_dir: 清洗数据保存目录，默认 CLEAN_DIR
        save: 是否保存清洗后数据

    Returns:
        dict: {'daily': clean_daily, 'basic': clean_basic, 'fina': clean_fina}
    """
    if raw_dir is None:
        raw_dir = RAW_DIR
    if clean_dir is None:
        clean_dir = CLEAN_DIR

    raw_dir = Path(raw_dir)
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📦 加载原始数据")
    print("=" * 60)

    # 加载 CSV 文件（使用全局路径变量）
    raw_daily = pd.read_csv(INPUT_DAILY)
    raw_basic = pd.read_csv(INPUT_BASIC)
    raw_fina  = pd.read_csv(INPUT_FINA)

    # 转换日期类型
    raw_daily['date'] = pd.to_datetime(raw_daily['date'])
    raw_basic['date'] = pd.to_datetime(raw_basic['date'])
    raw_fina['date']  = pd.to_datetime(raw_fina['date'])

    print(f"日线原始数据：{len(raw_daily):,} 行")
    print(f"估值原始数据：{len(raw_basic):,} 行")
    print(f"财务原始数据：{len(raw_fina):,} 行")

    # ========== 执行清洗 ==========
    print("\n" + "=" * 60)
    print("🧹 开始数据清洗")
    print("=" * 60)

    print("\n📈 清洗日线数据...")
    clean_daily = clean_daily_market_data(raw_daily, min_trade_days=60)

    print("\n📈 清洗估值数据...")
    clean_basic = clean_valuation_data(raw_basic)

    print("\n📈 清洗财务数据...")
    clean_fina = clean_financial_data(raw_fina)

    # ========== 数据对齐与合并 ==========
    merged_data = merge_and_align_data(clean_daily, clean_basic, clean_fina)

    # ========== 数据概览 ==========
    print("\n" + "=" * 60)
    print("📊 清洗后数据详细概览")
    print("=" * 60)

    # 日线概览
    print("\n📈 日线数据（clean_daily_market_data → clean_daily）")
    print(f"  行数：{len(clean_daily):,}")
    print(f"  列数：{len(clean_daily.columns)}")
    print(f"  股票数量：{clean_daily['symbol'].nunique()}")
    print(f"  日期范围：{clean_daily['date'].min().date()} ~ {clean_daily['date'].max().date()}")
    print(f"  收盘价区间：{clean_daily['close'].min():.2f} ~ {clean_daily['close'].max():.2f}")
    print(f"  缺失值：{clean_daily['daily_return'].isna().sum()}")

    # 估值概览
    print("\n📈 估值数据（clean_basic）")
    print(f"  行数：{len(clean_basic):,}")
    print(f"  列数：{len(clean_basic.columns)}")
    print(f"  股票数量：{clean_basic['symbol'].nunique()}")
    print(f"  日期范围：{clean_basic['date'].min().date()} ~ {clean_basic['date'].max().date()}")
    if 'pe_ttm' in clean_basic.columns:
        print(f"  PE_TTM 中位数：{clean_basic['pe_ttm'].median():.2f}")
    if 'pb' in clean_basic.columns:
        print(f"  PB 中位数：{clean_basic['pb'].median():.2f}")

    # 财务概览
    print("\n📈 财务数据（clean_fina）")
    print(f"  行数：{len(clean_fina):,}")
    print(f"  列数：{len(clean_fina.columns)}")
    print(f"  股票数量：{clean_fina['symbol'].nunique()}")
    print(f"  报告期范围：{clean_fina['date'].min().date()} ~ {clean_fina['date'].max().date()}")
    fina_counts = clean_fina.groupby('symbol').size()
    print(f"  每只股票财报数量：最小 {fina_counts.min()}，最大 {fina_counts.max()}，平均 {fina_counts.mean():.1f}")

    # ========== 保存 ==========
    if save:
        print("\n" + "=" * 60)
        print("💾 保存清洗后的数据")
        print("=" * 60)

        clean_daily.to_csv(OUTPUT_DAILY, index=False, encoding='utf-8-sig')
        clean_basic.to_csv(OUTPUT_BASIC, index=False, encoding='utf-8-sig')
        clean_fina.to_csv(OUTPUT_FINA, index=False, encoding='utf-8-sig')

        print(f"✅ 日线清洗数据：{len(clean_daily):,} 行 → {OUTPUT_DAILY}")
        print(f"✅ 估值清洗数据：{len(clean_basic):,} 行 → {OUTPUT_BASIC}")
        print(f"✅ 财务清洗数据：{len(clean_fina):,} 行 → {OUTPUT_FINA}")

        # 同时保存 Parquet 备份
        try:
            clean_daily.to_parquet(OUTPUT_DAILY.with_suffix('.parquet'), index=False)
            clean_basic.to_parquet(OUTPUT_BASIC.with_suffix('.parquet'), index=False)
            clean_fina.to_parquet(OUTPUT_FINA.with_suffix('.parquet'), index=False)
            print("\n✅ Parquet 备份已保存")
        except Exception as e:
            print(f"\n⚠️ Parquet 保存失败：{e}")

        # 保存合并后的对齐数据（供因子计算直接加载）
        merged_data.to_csv(OUTPUT_MERGED, index=False, encoding='utf-8-sig')
        try:
            merged_data.to_parquet(OUTPUT_MERGED.with_suffix('.parquet'), index=False)
            print(f"✅ 合并对齐数据：{len(merged_data):,} 行 → {OUTPUT_MERGED.with_suffix('.parquet')}")
        except Exception as e:
            print(f"⚠️ 合并数据 Parquet 保存失败：{e}")

    return {
        'daily': clean_daily,
        'basic': clean_basic,
        'fina': clean_fina,
        'merged': merged_data
    }


# ========================
# 6. 样本内外拆分
# ========================

def split_train_test(clean_data: dict, clean_dir=None,
                     train_start="2018-01-01", train_end="2023-12-31",
                     test_start="2024-01-01",  test_end="2026-06-30"):
    """
    将清洗后的数据按日期拆分为样本内（train）和样本外（test），分别保存。

    拆分规则：
      - 日线/估值/合并数据：按 date 列严格切分
      - 财务数据：按报告期 date 切分，保证测试集不会泄露训练集季报信息
      - 文件路径使用全局定义的 OUTPUT_*_IN / OUTPUT_*_OUT 变量

    Parameters:
        clean_data: run_data_cleaning 返回的 dict（含 daily, basic, fina, merged）
        clean_dir:  保存目录，默认 CLEAN_DIR
        train_start/train_end: 样本内日期范围
        test_start/test_end:   样本外日期范围
    """
    if clean_dir is None:
        clean_dir = CLEAN_DIR
    clean_dir = Path(clean_dir)

    train_start = pd.Timestamp(train_start)
    train_end   = pd.Timestamp(train_end)
    test_start  = pd.Timestamp(test_start)
    test_end    = pd.Timestamp(test_end)

    print("\n" + "=" * 60)
    print("✂️ 样本内外拆分")
    print("=" * 60)
    print(f"  样本内 train: {train_start.date()} ~ {train_end.date()}")
    print(f"  样本外 test:  {test_start.date()} ~ {test_end.date()}")

    # 文件映射：(data_key, output_in, output_out)
    file_map = [
        ('daily',  OUTPUT_DAILY_IN,  OUTPUT_DAILY_OUT),
        ('basic',  OUTPUT_BASIC_IN,  OUTPUT_BASIC_OUT),
        ('fina',   OUTPUT_FINA_IN,   OUTPUT_FINA_OUT),
        ('merged', OUTPUT_MERGED_IN, OUTPUT_MERGED_OUT),
    ]

    for key, out_in, out_out in file_map:
        df = clean_data.get(key)
        if df is None or df.empty:
            print(f"  ⚠️ {key} 数据为空，跳过")
            continue

        train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
        test_df  = df[(df['date'] >= test_start)  & (df['date'] <= test_end)].copy()

        print(f"\n  📂 {key}: 全量 {len(df):,}行 → 样本内 {len(train_df):,}行 | 样本外 {len(test_df):,}行")

        # 保存 CSV
        train_df.to_csv(out_in, index=False, encoding='utf-8-sig')
        test_df.to_csv(out_out, index=False, encoding='utf-8-sig')

        # 保存 Parquet
        try:
            train_df.to_parquet(out_in.with_suffix('.parquet'), index=False)
            test_df.to_parquet(out_out.with_suffix('.parquet'), index=False)
        except Exception as e:
            print(f"    ⚠️ Parquet 保存失败：{e}")

    # ---- 同时拆分原始指数数据（供回测基准使用） ----
    if INPUT_INDEX.exists():
        index_df = pd.read_csv(INPUT_INDEX)
        index_df['date'] = pd.to_datetime(index_df['date'])
        idx_train = index_df[(index_df['date'] >= train_start) & (index_df['date'] <= train_end)]
        idx_test  = index_df[(index_df['date'] >= test_start)  & (index_df['date'] <= test_end)]
        print(f"\n  📂 index: 全量 {len(index_df):,}行 → 样本内 {len(idx_train):,}行 | 样本外 {len(idx_test):,}行")

        idx_train.to_csv(OUTPUT_INDEX_IN, index=False, encoding='utf-8-sig')
        idx_test.to_csv(OUTPUT_INDEX_OUT, index=False, encoding='utf-8-sig')
        try:
            idx_train.to_parquet(OUTPUT_INDEX_IN.with_suffix('.parquet'), index=False)
            idx_test.to_parquet(OUTPUT_INDEX_OUT.with_suffix('.parquet'), index=False)
        except Exception as e:
            print(f"    ⚠️ Parquet 保存失败：{e}")

    print(f"\n✅ 样本内外拆分完成，文件保存至：{clean_dir}")


# ========================
# 7. 工具函数：数据摘要
# ========================

def print_data_summary(clean_data: dict):
    """打印清洗后数据摘要"""
    print("\n" + "=" * 60)
    print("📊 清洗后数据摘要")
    print("=" * 60)

    for key, df in clean_data.items():
        if df is None or df.empty:
            print(f"  ❌ {key}: 数据为空")
            continue
        print(f"\n  📈 {key}:")
        print(f"     行数：{len(df):,}")
        print(f"     列数：{len(df.columns)}")
        print(f"     列名：{list(df.columns)}")


# ========================
# 8. 主执行入口
# ========================

if __name__ == "__main__":
    print("=" * 60)
    print("🧹 数据清洗模块 - 沪深300多因子选股")
    print("=" * 60)

    # 执行清洗流程
    clean_data = run_data_cleaning(
        raw_dir=RAW_DIR,
        clean_dir=CLEAN_DIR,
        save=True
    )

    # 打印摘要
    print_data_summary(clean_data)

    # ========== 样本内外拆分 ==========
    split_train_test(
        clean_data,
        clean_dir=CLEAN_DIR,
        train_start="2018-01-01",
        train_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2026-06-30"
    )

    print(f"\n{'='*60}")
    print("✅ 数据清洗 + 样本内外拆分全部完成！")
    print(f"{'='*60}")
