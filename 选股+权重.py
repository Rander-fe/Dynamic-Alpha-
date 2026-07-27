"""
持仓管理模块
==================================================
功能：
  1. 根据因子评分（rolling_score）生成月度调仓指令（monthly_positions）
  2. 回测中执行调仓（统一再平衡，支持权重分配）

输入：因子评分文件（Parquet格式，含 composite_score）
输出：月度持仓文件（Parquet/CSV，含 trade_date, ts_code, weight）

调仓执行：rebalance_portfolio 函数，用于回测引擎中
==================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 路径配置
# ============================================================
PROJECT_ROOT = Path(r"C:/Users/haoran/Desktop/动态因子选股")
DATA_DIR     = PROJECT_ROOT / "data"
FACTOR_DIR   = DATA_DIR / "factors"
SELECTION_DIR = DATA_DIR / "selection"

SELECTION_DIR.mkdir(parents=True, exist_ok=True)

# 样本类型列表
SAMPLE_TYPES = ["insample", "outsample"]

def get_score_file(sample_type):
    return FACTOR_DIR / f"rolling_score_{sample_type}.parquet"

def get_selection_log_file(sample_type):
    return FACTOR_DIR / f"rolling_selection_log_{sample_type}.csv"

# ============================================================
# 1. 核心参数（可根据需要调整）
# ============================================================
TOP_N           = 30          # 每月选股数
TURNOVER_BUFFER = 10          # 换手缓冲（上月持仓在 TOP_N+buffer 内保留）
COST_RATE       = 0.001       # 交易成本（双边，千分之一）


# ============================================================
# 2. 辅助函数（供调仓使用）
# ============================================================
def is_traded(date, sym, open_pivot):
    """判断股票在当天是否正常交易（开盘价有效）"""
    if sym not in open_pivot.columns:
        return False
    price = open_pivot.loc[date, sym]
    return pd.notna(price) and price > 0


def get_next_trade_date(date, all_dates):
    """获取 date 之后的下一个交易日（仅用于回测）"""
    pos = all_dates.get_loc(date)
    next_idx = pos + 1
    if next_idx < len(all_dates):
        return all_dates[next_idx]
    return None


# ============================================================
# 3. 选股生成：从因子评分到月度持仓
# ============================================================
def generate_monthly_positions(score_file, log_file=None,
                               top_n=TOP_N, buffer=TURNOVER_BUFFER,
                               start_date=None, end_date=None,
                               output_dir=None):
    """
    从因子评分文件生成月度持仓指令

    参数
    ----
    score_file : Path 或 str
        因子评分 Parquet 文件路径（必须包含 date, symbol, composite_score）
    log_file : Path 或 str, 可选
        筛选日志 CSV 文件路径（用于识别零因子月）
    top_n : int
        每月选股数量
    buffer : int
        换手缓冲（上月持仓在 top_n+buffer 内保留）
    start_date, end_date : str, 可选
        日期过滤范围，如 '2018-01-01'
    output_dir : Path 或 str, 可选
        输出目录，默认为 score_file.parent.parent / 'selection'

    返回
    ----
    pos_df : pd.DataFrame
        月度持仓表，包含 trade_date, year_month, ts_code, weight
    ranking_df : pd.DataFrame
        月度排名表（含 rank, has_score）
    log_df : pd.DataFrame
        处理日志
    """
    score_file = Path(score_file)
    if output_dir is None:
        output_dir = score_file.parent.parent / 'selection'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载评分 ----
    print(f"[LOAD] 评分文件：{score_file}")
    score_df = pd.read_parquet(score_file)
    score_df['date'] = pd.to_datetime(score_df['date'])
    if start_date:
        score_df = score_df[score_df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        score_df = score_df[score_df['date'] <= pd.to_datetime(end_date)]
    print(f"   行数：{len(score_df):,}，日期范围：{score_df['date'].min().date()} ~ {score_df['date'].max().date()}")

    # ---- 加载日志（可选） ----
    log_df = None
    if log_file and Path(log_file).exists():
        log_df = pd.read_csv(log_file)
        log_df.columns = log_df.columns.str.strip().str.lower()
        print(f"[LOAD] 日志文件：{len(log_df)} 个月")

    # ---- 提取月末调仓日 ----
    df = score_df.copy()
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    last_dates = df.groupby('year_month')['date'].max().reset_index()
    rebalance = df.merge(last_dates, on=['year_month', 'date'], how='inner')
    rebalance = rebalance.sort_values(['date', 'symbol']).reset_index(drop=True)
    print(f"[REBALANCE] 调仓日：{rebalance['year_month'].nunique()} 个月，{len(rebalance):,} 行")

    # ---- 月度排名 ----
    ranking = rebalance.copy()
    ranking['rank'] = ranking.groupby('date')['composite_score'].rank(ascending=False, method='first')
    ranking['has_score'] = ranking['composite_score'].notna()
    ranking = ranking[['date', 'year_month', 'symbol', 'composite_score', 'rank', 'has_score']]

    # ---- 缓冲选股 ----
    def select_with_buffer(month_df, prev_holdings):
        if month_df['composite_score'].isna().all():
            return None
        df_sorted = month_df.dropna(subset=['composite_score']).sort_values('composite_score', ascending=False)
        if len(df_sorted) == 0:
            return None
        if prev_holdings is None or len(prev_holdings) == 0:
            return df_sorted.head(top_n)['symbol'].tolist()
        candidates = df_sorted.head(top_n + buffer)
        locked = set(candidates.loc[candidates['symbol'].isin(prev_holdings), 'symbol'])
        selected = set(locked)
        for _, row in candidates.iterrows():
            if len(selected) >= top_n:
                break
            if row['symbol'] not in selected:
                selected.add(row['symbol'])
        if len(selected) < top_n:
            remaining = df_sorted[~df_sorted['symbol'].isin(selected)]
            extra = remaining.head(top_n - len(selected))
            selected.update(extra['symbol'].tolist())
        return list(selected)

    pos_list = []
    prev = None
    prev_positions = None
    n_hold = 0
    n_csi300 = 0

    for ym, group in ranking.groupby('year_month'):
        selected = select_with_buffer(group, prev)
        trade_date = group['date'].max()

        if selected is None:
            if prev_positions is not None and len(prev_positions) > 0:
                for p in prev_positions:
                    pos_list.append({'trade_date': trade_date, 'year_month': ym,
                                     'ts_code': p['ts_code'], 'weight': p['weight']})
                n_hold += 1
                continue
            else:
                pos_list.append({'trade_date': trade_date, 'year_month': ym,
                                 'ts_code': 'CSI300', 'weight': 1.0})
                n_csi300 += 1
                prev = None
                prev_positions = None
                continue

        # Softmax 权重
        sel_scores = group[group['symbol'].isin(selected)][['symbol', 'composite_score']].drop_duplicates('symbol')
        scores = sel_scores['composite_score'].values
        exp_scores = np.exp(scores - scores.max())
        weight_arr = exp_scores / exp_scores.sum()
        weight_map = dict(zip(sel_scores['symbol'], weight_arr))

        for sym in selected:
            pos_list.append({'trade_date': trade_date, 'year_month': ym,
                             'ts_code': sym, 'weight': weight_map[sym]})

        prev = set(selected)
        prev_positions = [{'ts_code': s, 'weight': weight_map[s]} for s in selected]

    pos_df = pd.DataFrame(pos_list)
    print(f"[OK] 持仓生成：{len(pos_df):,} 条，覆盖 {pos_df['year_month'].nunique()} 个月")
    if n_csi300 > 0:
        print(f"   零因子/无历史回退CSI300：{n_csi300} 个月")

    # ---- 保存 ----
    ranking_file = output_dir / f"monthly_rankings_{score_file.stem.replace('rolling_score_','')}.parquet"
    pos_file = output_dir / f"monthly_positions_{score_file.stem.replace('rolling_score_','')}.parquet"
    ranking.to_parquet(ranking_file, index=False)
    pos_df.to_parquet(pos_file, index=False)
    print(f"[SAVE] 排名：{ranking_file}")
    print(f"[SAVE] 持仓：{pos_file}")

    # 同时保存 CSV 方便查看
    ranking.to_csv(ranking_file.with_suffix('.csv'), index=False, encoding='utf-8-sig')
    pos_df.to_csv(pos_file.with_suffix('.csv'), index=False, encoding='utf-8-sig')

    return pos_df, ranking, log_df


# ============================================================
# 4. 回测调仓执行（统一再平衡）
# ============================================================
def rebalance_portfolio(holdings, cash, day_positions, position_ratio,
                        trade_date, open_pivot, cost_rate=COST_RATE):
    """
    执行调仓日的全部交易：
      - 卖出离场股票（old - new）
      - 统一再平衡：按 day_positions 中的 weight 列分配目标市值
      返回：更新后的 holdings, cash

    参数
    ----
    holdings : dict {symbol: shares}
    cash : float
    day_positions : pd.DataFrame
        当月目标持仓表，必须包含 ts_code, weight 列
    position_ratio : float
        目标总仓位（0~1）
    trade_date : pd.Timestamp
        调仓日（执行日）
    open_pivot : pd.DataFrame
        开盘价矩阵（date × symbol）
    cost_rate : float
        交易成本率

    返回
    ----
    holdings, cash
    """
    new_stocks = set(day_positions['ts_code'].tolist()) if not day_positions.empty else set()
    old_stocks = set(holdings.keys())

    # ---- 1. 卖出离场股 ----
    sell_stocks = old_stocks - new_stocks
    for sym in list(sell_stocks):
        shares = holdings.get(sym, 0)
        if shares <= 0:
            continue
        if not is_traded(trade_date, sym, open_pivot):
            continue
        price = open_pivot.loc[trade_date, sym]
        revenue = shares * price * (1 - cost_rate)
        cash += revenue
        del holdings[sym]

    # ---- 2. 统一再平衡（按权重） ----
    if not day_positions.empty:
        # 构建可交易标的列表
        all_targets = []
        for _, row in day_positions.iterrows():
            sym = row['ts_code']
            if not is_traded(trade_date, sym, open_pivot):
                continue  # 停牌股暂不处理，保留原仓位（如果有）
            price = open_pivot.loc[trade_date, sym]
            cur_shares = holdings.get(sym, 0)
            all_targets.append((sym, cur_shares > 0, cur_shares, price, row['weight']))

        if all_targets:
            # 计算总资产和目标权益
            holdings_value = sum(shares * price for _, held, shares, price, _ in all_targets if held)
            total_value = cash + holdings_value
            target_equity = total_value * position_ratio

            # 按权重分配目标市值
            total_weight = sum(w for _, _, _, _, w in all_targets)
            if total_weight > 0:
                for sym, is_held, cur_shares, price, w in all_targets:
                    target_value = target_equity * (w / total_weight)
                    target_shares = int(target_value / price) if price > 0 else 0
                    if is_held:
                        if target_shares < cur_shares:
                            sell_shares = cur_shares - target_shares
                            cash += sell_shares * price * (1 - cost_rate)
                            if target_shares > 0:
                                holdings[sym] = target_shares
                            else:
                                del holdings[sym]
                        elif target_shares > cur_shares:
                            buy_shares = target_shares - cur_shares
                            cost = buy_shares * price * (1 + cost_rate)
                            if cost <= cash:
                                cash -= cost
                                holdings[sym] = target_shares
                        # else: 持股不变
                    else:
                        if target_shares > 0:
                            cost = target_shares * price * (1 + cost_rate)
                            if cost <= cash:
                                cash -= cost
                                holdings[sym] = target_shares
    return holdings, cash


# ============================================================
# 5. 主入口：样本内 + 样本外
# ============================================================
def main():
    """依次对样本内、样本外生成月度持仓指令"""
    all_pos = {}
    for st in SAMPLE_TYPES:
        print("\n" + "=" * 60)
        print(f"📊 选股+权重 — {st.upper()}")
        print("=" * 60)
        score_file = get_score_file(st)
        log_file = get_selection_log_file(st)

        if not score_file.exists():
            print(f"⚠️ [{st}] 评分文件不存在：{score_file}，跳过")
            continue

        pos_df, ranking_df, log_df = generate_monthly_positions(
            score_file=score_file,
            log_file=log_file,
            top_n=TOP_N,
            buffer=TURNOVER_BUFFER,
            output_dir=SELECTION_DIR
        )
        all_pos[st] = pos_df
        print(f"[DONE] {st} 持仓指令生成完成")

    if len(all_pos) == 0:
        print("⚠️ 无任何样本数据可处理")
        return

    # ---- 汇总 ---- 
    for st, pos_df in all_pos.items():
        print(f"\n{st}: {len(pos_df)} 条持仓，"
              f"{pos_df['trade_date'].nunique()} 个调仓日，"
              f"{pos_df['ts_code'].nunique()} 只标的")

    print("\n[OK] 全部完成！")


if __name__ == "__main__":
    main()