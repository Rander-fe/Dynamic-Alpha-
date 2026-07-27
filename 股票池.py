# ============================================================
# 📦 股票池配置 —— 沪深300成分股动态获取
# ============================================================
# 功能：
#   1. 配置 Tushare Pro Token 与环境
#   2. 设置项目目录结构
#   3. 动态获取沪深300成分股（2018-01-01 ~ 2026-06-30）
#   4. 保存股票池信息到文件
# ============================================================

import os
import time
import json
import warnings
from pathlib import Path

import pandas as pd
import tushare as ts

warnings.filterwarnings("ignore")

# ========================
# 1. Tushare Pro Token 配置
# ========================
# 从环境变量读取（推荐），如无则使用默认值
TUSHARE_TOKEN = os.environ.get(
    "TUSHARE_TOKEN",
    "ba3ef3862d622efeb8a0214b8ea6dfac00809e234af80b2c40035e8c"
)

if not TUSHARE_TOKEN or TUSHARE_TOKEN == "your_token_here":
    print("⚠️ 请先设置 Tushare Pro Token！")
    print("   注册地址：https://tushare.pro/register")
    print("   设置方式：$env:TUSHARE_TOKEN='your_token'")
    raise ValueError("Tushare Token 未配置")

pro = ts.pro_api(TUSHARE_TOKEN)
print("✅ Tushare Pro 初始化成功")

# ========================
# 2. 项目路径配置
# ========================
PROJECT_ROOT = Path.cwd()  # 项目根目录
DATA_DIR = PROJECT_ROOT / "data"            # 数据根目录
RAW_DIR = DATA_DIR / "raw"                  # 原始数据
CLEAN_DIR = DATA_DIR / "clean"       # 清洗后数据
FACTOR_DIR = DATA_DIR / "factors"           # 因子数据
MODEL_DIR = PROJECT_ROOT / "models"         # 模型保存
CONFIG_DIR = PROJECT_ROOT / "config"        # 配置文件
LOG_DIR = PROJECT_ROOT / "logs"             # 日志

# 创建目录
for d in [DATA_DIR, RAW_DIR, CLEAN_DIR, FACTOR_DIR, MODEL_DIR, CONFIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"📁 项目根目录：{PROJECT_ROOT}")
print(f"📁 原始数据：{RAW_DIR}")
print(f"📁 清洗数据：{CLEAN_DIR}")
print(f"📁 因子数据：{FACTOR_DIR}")

# ========================
# 3. 时间范围配置
# ========================
# 全时段：2018-01-01 ~ 2026-06-30
START_DATE = "20180101"    # 数据起始日期
END_DATE = "20260630"      # 数据结束日期

# 样本内（训练/调优期）
TRAIN_START = "2018-01-01"
TRAIN_END = "2024-06-30"
# 样本外（验证期）
TEST_START = "2024-07-01"
TEST_END = "2026-06-30"

print(f"📅 数据时间范围：{START_DATE} ~ {END_DATE}")
print(f"📅 样本内：{TRAIN_START} ~ {TRAIN_END}")
print(f"📅 样本外：{TEST_START} ~ {TEST_END}")

# ========================
# 4. 回测参数配置
# ========================
TOP_N = 30                # 持仓数量
TURNOVER_BUFFER = 10      # 换手缓冲区
COST_RATE = 0.001         # 单边交易成本（0.1%）

# ========================
# 5. 动态获取沪深300成分股
# ========================

def get_csi300_constituents(pro, save_path=None):
    """
    通过 Tushare index_weight 接口获取沪深300最新成分股列表。

    Parameters:
        pro: Tushare Pro API 实例
        save_path: 保存路径（可选），支持 CSV / Parquet

    Returns:
        list[dict]: 每项包含 symbol（6位代码）、name（股票名称）、list_date（上市日期）
    """
    print("\n📡 正在获取沪深300成分股...")

    # 获取指数权重数据（含多个调仓日快照）
    df = pro.index_weight(index_code='000300.SH')

    if df is None or df.empty:
        raise ValueError("❌ 获取沪深300成分股失败，请检查Tushare Token权限")

    print(f"   index_weight 返回 {len(df)} 条记录（含多个调仓日）")

    # 取最新一个调仓日的成分股
    latest_date = df['trade_date'].max()
    df_latest = df[df['trade_date'] == latest_date]
    print(f"   取最新调仓日：{latest_date}，共 {len(df_latest)} 只成分股")

    # con_code 格式如 '000001.SZ' → 提取6位数字代码
    symbols = df_latest['con_code'].str.split('.').str[0].tolist()

    # 批量获取股票名称和信息
    name_map = {}
    try:
        ts_codes = [
            f"{s}.SH" if s.startswith('6') else f"{s}.SZ"
            for s in symbols
        ]
        # 分批次查询（每批最多100条）
        for i in range(0, len(ts_codes), 100):
            batch = ts_codes[i:i + 100]
            basic_df = pro.stock_basic(
                ts_code=','.join(batch),
                fields='ts_code,name,list_date,market,industry'
            )
            if basic_df is not None and not basic_df.empty:
                for _, row in basic_df.iterrows():
                    sym = row['ts_code'].split('.')[0]
                    name_map[sym] = {
                        'name': row['name'],
                        'list_date': row.get('list_date', ''),
                        'market': row.get('market', ''),
                        'industry': row.get('industry', ''),
                    }
            time.sleep(0.3)  # 控制请求频率
    except Exception as e:
        print(f"   ⚠️ 获取股票信息失败：{e}")

    # 组装股票池
    pool = []
    for symbol in symbols:
        info = name_map.get(symbol, {})
        pool.append({
            "symbol": symbol,
            "name": info.get('name', ''),
            "list_date": info.get('list_date', ''),
            "market": info.get('market', ''),
            "industry": info.get('industry', ''),
        })

    print(f"\n✅ 沪深300成分股获取完成：{len(pool)} 只")

    # 保存到文件
    if save_path:
        pool_df = pd.DataFrame(pool)
        save_path = Path(save_path)
        if save_path.suffix == '.csv':
            pool_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        elif save_path.suffix == '.parquet':
            pool_df.to_parquet(save_path, index=False)
        elif save_path.suffix == '.json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(pool, f, ensure_ascii=False, indent=2)
        print(f"   已保存到：{save_path}")

    return pool


# ========================
# 6. 执行获取并保存股票池
# ========================
STOCK_POOL_FILE = RAW_DIR / "csi300_stock_pool.csv"
STOCK_POOL_JSON = CONFIG_DIR / "stock_pool_config.json"

# 获取股票池
STOCK_POOL = get_csi300_constituents(pro, save_path=STOCK_POOL_FILE)

# 同时保存 JSON 格式（便于程序读取）
pool_df = pd.DataFrame(STOCK_POOL)
pool_df.to_json(STOCK_POOL_JSON, orient='records', force_ascii=False)
print(f"   已保存配置到：{STOCK_POOL_JSON}")

# ========================
# 7. 打印股票池概览
# ========================
print("\n" + "=" * 60)
print("📊 股票池概览")
print("=" * 60)
print(f"  总股票数：{len(STOCK_POOL)}")

# 行业分布
if STOCK_POOL[0].get('industry'):
    industry_counts = {}
    for s in STOCK_POOL:
        ind = s.get('industry', '未知')
        industry_counts[ind] = industry_counts.get(ind, 0) + 1

    print(f"\n  行业分布（前10）：")
    for ind, cnt in sorted(industry_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {ind}: {cnt} 只")

# 市场分布
market_counts = {}
for s in STOCK_POOL:
    mkt = s.get('market', '未知')
    market_counts[mkt] = market_counts.get(mkt, 0) + 1

print(f"\n  市场分布：")
for mkt, cnt in sorted(market_counts.items(), key=lambda x: -x[1]):
    print(f"    {mkt}: {cnt} 只")

# 预览前10只
print(f"\n  前10只成分股：")
for i, s in enumerate(STOCK_POOL[:10], 1):
    print(f"    {i:2d}. {s['symbol']} {s['name']}  上市日: {s.get('list_date', 'N/A')}  行业: {s.get('industry', 'N/A')}")

print("\n" + "=" * 60)
print("✅ 股票池配置完成！")
print("=" * 60)
