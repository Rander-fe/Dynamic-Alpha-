# ============================================================
# 📡 数据获取 —— 批量获取沪深300股票数据
# ============================================================
# 功能：
#   1. 导入依赖与路径配置
#   2. 股票代码转换工具函数
#   3. 批量获取：日线行情 + 每日估值 + 全部历史季度财务指标
#   4. 获取沪深300指数数据
#   5. 保存原始数据到文件（CSV + Parquet）
# ============================================================
# 时间范围：2018-01-01 ~ 2026-06-30
# 数据来源：Tushare Pro
# ============================================================

import os
import time
import warnings
from pathlib import Path

import pandas as pd
import tushare as ts

warnings.filterwarnings("ignore")

# ========================
# 1. 环境配置
# ========================
TUSHARE_TOKEN = os.environ.get(
    "TUSHARE_TOKEN",
    "ba3ef3862d622efeb8a0214b8ea6dfac00809e234af80b2c40035e8c"
)

if not TUSHARE_TOKEN or TUSHARE_TOKEN == "your_token_here":
    print("⚠️ 请先设置 Tushare Pro Token！")
    raise ValueError("Tushare Token 未配置")

pro = ts.pro_api(TUSHARE_TOKEN)
print("✅ Tushare Pro 初始化成功")

# ========================
# 2. 路径配置
# ========================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
FACTOR_DIR = DATA_DIR / "factors"
LOG_DIR = PROJECT_ROOT / "logs"

for d in [RAW_DIR, CLEAN_DIR, FACTOR_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"📁 原始数据目录：{RAW_DIR}")
print(f"📁 清洗数据目录：{CLEAN_DIR}")

# ========================
# 3. 时间范围
# ========================
START_DATE = "20180101"    # 2018-01-01
END_DATE   = "20260630"    # 2026-06-30

print(f"📅 数据获取时间范围：{START_DATE} ~ {END_DATE}")

# ========================
# 4. 股票代码转换工具
# ========================

def symbol_to_ts_code(symbol: str) -> str:
    """
    将 6 位数字代码转为 Tushare 格式。
    上交所 6xxxxx → .SH  |  深交所 0/2/3xxxxx → .SZ
    """
    symbol = str(symbol).zfill(6)
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    elif symbol.startswith(("0", "2", "3")):
        return f"{symbol}.SZ"
    else:
        raise ValueError(f"无法识别的股票代码：{symbol}")


# ========================
# 5. 核心函数：批量获取数据
# ========================

def fetch_all_stocks(pool, start=START_DATE, end=END_DATE, sleep=0.6,
                     get_fundamental=True, get_index=True):
    """
    批量获取股票数据：日线行情 + 每日估值 + 全部历史季度财务指标

    Parameters:
        pool: list[dict]，每项含 symbol, name
        start: str, 开始日期 YYYYMMDD
        end: str, 结束日期 YYYYMMDD
        sleep: float, 每只股票请求间隔（秒）
        get_fundamental: bool, 是否获取估值+财务数据
        get_index: bool, 是否获取沪深300指数

    Returns:
        dict: {
            'daily': DataFrame 日线行情,
            'basic': DataFrame 每日估值,
            'fina':  DataFrame 全部历史季报,
            'index': DataFrame 指数行情,
            'success': list[dict],
            'failed': list[dict]
        }
    """
    all_daily = []
    all_basic = []
    all_fina = []
    success_list = []
    failed_list = []

    total = len(pool)
    print(f"\n{'='*60}")
    print(f"📡 开始获取 {total} 只股票数据...")
    print(f"{'='*60}")

    for idx, item in enumerate(pool, 1):
        symbol = item["symbol"]
        name = item["name"]
        ts_code = symbol_to_ts_code(symbol)

        try:
            # ========== 1. 日线行情（前复权） ==========
            df_daily = pro.daily(
                ts_code=ts_code,
                start_date=start,
                end_date=end,
                adj='qfq'
            )
            if df_daily is None or df_daily.empty:
                failed_list.append({"symbol": symbol, "name": name, "error": "日线空数据"})
                print(f"[{idx:2d}/{total}] ❌ {symbol} {name} — 日线空")
                continue

            # 标准化日线
            df_daily = df_daily.rename(columns={
                "trade_date": "date",
            })
            if "pct_chg" in df_daily.columns:
                df_daily["pct_chg"] = df_daily["pct_chg"] / 100  # 转小数

            df_daily["symbol"] = str(symbol).zfill(6)
            df_daily["name"] = name
            df_daily["date"] = pd.to_datetime(df_daily["date"], format="%Y%m%d")
            df_daily = df_daily.sort_values("date").reset_index(drop=True)
            all_daily.append(df_daily)

            # ========== 2. 每日估值（市值、PE/PB等） ==========
            df_basic = None
            if get_fundamental:
                df_basic = pro.daily_basic(
                    ts_code=ts_code,
                    start_date=start,
                    end_date=end,
                    fields='trade_date,total_mv,float_share,pe_ttm,pb,ps_ttm,turnover_rate'
                )
                if df_basic is not None and not df_basic.empty:
                    df_basic = df_basic.rename(columns={"trade_date": "date"})
                    df_basic["symbol"] = str(symbol).zfill(6)
                    df_basic["date"] = pd.to_datetime(df_basic["date"], format="%Y%m%d")
                    all_basic.append(df_basic)

                # ========== 3. 全部历史季报（不限时间） ==========
                df_fina = pro.fina_indicator(
                    ts_code=ts_code,
                    fields='end_date,roe,netprofit_yoy,revenue_yoy,'
                           'grossprofit_margin,debt_to_assets,net_assets,'
                           'net_profit,revenue'
                )
                if df_fina is not None and not df_fina.empty:
                    df_fina = df_fina.rename(columns={"end_date": "date"})
                    df_fina["symbol"] = str(symbol).zfill(6)
                    df_fina["date"] = pd.to_datetime(df_fina["date"])
                    df_fina = df_fina.sort_values("date").reset_index(drop=True)
                    all_fina.append(df_fina)

            # 记录成功
            rec = {
                "symbol": str(symbol).zfill(6),
                "name": name,
                "daily_rows": len(df_daily),
            }
            if get_fundamental:
                rec["basic_rows"] = len(df_basic) if df_basic is not None else 0
                rec["fina_rows"] = len(df_fina) if 'df_fina' in dir() and df_fina is not None else 0
            success_list.append(rec)

            # 打印进度
            msg = f"[{idx:2d}/{total}] ✅ {symbol} {name} — 日线{len(df_daily)}行"
            if get_fundamental:
                msg += f" | 估值{rec.get('basic_rows', 0)}行"
                msg += f" | 财务{rec.get('fina_rows', 0)}行"
            print(msg)

        except Exception as e:
            failed_list.append({"symbol": symbol, "name": name, "error": str(e)})
            print(f"[{idx:2d}/{total}] ❌ {symbol} {name} — {e}")

        # 频率控制
        if idx < total:
            time.sleep(sleep)

    # ---- 合并 DataFrame ----
    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    basic_df = pd.concat(all_basic, ignore_index=True) if all_basic else pd.DataFrame()
    fina_df = pd.concat(all_fina, ignore_index=True) if all_fina else pd.DataFrame()

    # ---- 获取沪深300指数 ----
    index_df = pd.DataFrame()
    if get_index:
        try:
            index_df = pro.index_daily(ts_code='000300.SH', start_date=start, end_date=end)
            if index_df is not None and not index_df.empty:
                index_df = index_df.rename(columns={
                    "trade_date": "date",
                    "close": "index_close"
                })
                index_df["date"] = pd.to_datetime(index_df["date"], format="%Y%m%d")
                index_df = index_df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ 获取指数失败：{e}")

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"📊 数据获取完成！")
    print(f"{'='*60}")
    print(f"  成功：{len(success_list)} 只")
    print(f"  失败：{len(failed_list)} 只")
    if not daily_df.empty:
        print(f"  日线数据：{len(daily_df):,} 行，{daily_df['symbol'].nunique()} 只股票")
    if not basic_df.empty:
        print(f"  估值数据：{len(basic_df):,} 行，{basic_df['symbol'].nunique()} 只股票")
    if not fina_df.empty:
        print(f"  财务数据：{len(fina_df):,} 行，{fina_df['symbol'].nunique()} 只股票")
    if not index_df.empty:
        print(f"  指数数据：{len(index_df):,} 行")

    return {
        'daily': daily_df,
        'basic': basic_df,
        'fina': fina_df,
        'index': index_df,
        'success': success_list,
        'failed': failed_list
    }


# ========================
# 6. 保存数据到文件
# ========================

def save_raw_data(data_dict, raw_dir=None):
    """
    将获取到的原始数据保存为 CSV 和 Parquet 格式。

    Parameters:
        data_dict: fetch_all_stocks 返回的字典
        raw_dir: 保存目录，默认 RAW_DIR
    """
    if raw_dir is None:
        raw_dir = RAW_DIR
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 正在保存原始数据到：{raw_dir}")
    print("-" * 50)

    # CSV 保存
    csv_files = {}
    for name_key in ['daily', 'basic', 'fina', 'index']:
        df = data_dict.get(name_key)
        if df is not None and not df.empty:
            fname = f"stock_{name_key}_raw.csv" if name_key != 'index' else "index_000300_raw.csv"
            fpath = raw_dir / fname
            df.to_csv(fpath, index=False, encoding='utf-8-sig')
            csv_files[name_key] = fpath
            print(f"  ✅ CSV: {fname} ({len(df):,} 行)")

    # Parquet 保存（可选）
    try:
        for name_key in ['daily', 'basic', 'fina', 'index']:
            df = data_dict.get(name_key)
            if df is not None and not df.empty:
                fname = f"stock_{name_key}_raw.parquet" if name_key != 'index' else "index_000300_raw.parquet"
                fpath = raw_dir / fname
                df.to_parquet(fpath, index=False)
                print(f"  ✅ Parquet: {fname}")
        print("\n✅ Parquet 备份保存成功")
    except Exception as e:
        print(f"\n⚠️ Parquet 保存失败（需安装 pyarrow）：{e}")
        print("    CSV 已正常保存，可直接使用。")

    return csv_files


# ========================
# 7. 主执行流程
# ========================

if __name__ == "__main__":
    print("=" * 60)
    print("📡 数据获取模块 - 沪深300多因子选股")
    print("=" * 60)

    # 加载股票池
    from pathlib import Path as _Path
    pool_file = PROJECT_ROOT / "config" / "stock_pool_config.json"
    if pool_file.exists():
        import json
        with open(pool_file, 'r', encoding='utf-8') as f:
            STOCK_POOL = json.load(f)
        print(f"\n✅ 从 {pool_file} 加载股票池：{len(STOCK_POOL)} 只股票")
    else:
        print("\n⚠️ 未找到股票池配置文件，请先运行「股票池」模块")
        # 从 CSV 尝试加载
        csv_pool = RAW_DIR / "csi300_stock_pool.csv"
        if csv_pool.exists():
            pool_df = pd.read_csv(csv_pool)
            STOCK_POOL = pool_df.to_dict('records')
            print(f"   从 {csv_pool} 加载股票池：{len(STOCK_POOL)} 只股票")
        else:
            raise FileNotFoundError("请先运行「股票池」模块获取沪深300成分股")

    # 执行数据获取
    data_dict = fetch_all_stocks(
        pool=STOCK_POOL,
        start=START_DATE,
        end=END_DATE,
        sleep=0.6,
        get_fundamental=True,
        get_index=True
    )

    # 保存数据
    save_raw_data(data_dict, RAW_DIR)

    print(f"\n{'='*60}")
    print("✅ 数据获取全部完成！")
    print(f"{'='*60}")
