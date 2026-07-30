# -*- coding: utf-8 -*-
"""
回测引擎（整合版）
==================================================
功能：
  - 读取月度调仓指令（含权重）
  - T+1 开盘执行，统一再平衡
  - 动态仓位管理（回撤阶梯 + 波动率目标）
  - 输出净值曲线和绩效报告
  - 支持样本内 / 样本外双回测

输入：data/selection/monthly_positions_*.parquet
      data/clean/stock_daily_clean_*.parquet
      data/raw/index_000300_raw_*.parquet
输出：data/selection/backtest_nav_*.parquet
      data/selection/performance_summary_*.csv
      净值曲线图
==================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 调仓执行函数（内联，来自 选股+权重.py）
# ============================================================
def rebalance_portfolio(holdings, cash, day_positions, position_ratio,
                        trade_date, open_pivot, cost_rate=0.001):
    """
    执行调仓日全部交易：卖出离场股 + 按权重统一再平衡
    返回：更新后的 holdings, cash
    """
    new_stocks = set(day_positions['ts_code'].tolist()) if not day_positions.empty else set()
    old_stocks = set(holdings.keys())

    # -- 1. 卖出离场股 --
    for sym in list(old_stocks - new_stocks):
        shares = holdings.get(sym, 0)
        if shares <= 0:
            continue
        if not is_traded(trade_date, sym, open_pivot):
            continue
        price = open_pivot.loc[trade_date, sym]
        cash += shares * price * (1 - cost_rate)
        del holdings[sym]

    # -- 2. 统一再平衡（按权重） --
    if day_positions.empty:
        return holdings, cash

    all_targets = []
    for _, row in day_positions.iterrows():
        sym = row['ts_code']
        if not is_traded(trade_date, sym, open_pivot):
            continue
        price = open_pivot.loc[trade_date, sym]
        cur_shares = holdings.get(sym, 0)
        all_targets.append((sym, cur_shares > 0, cur_shares, price, row['weight']))

    if not all_targets:
        return holdings, cash

    holdings_value = sum(shares * price for _, held, shares, price, _ in all_targets if held)
    total_value = cash + holdings_value
    target_equity = total_value * position_ratio

    total_weight = sum(w for _, _, _, _, w in all_targets)
    if total_weight <= 0:
        return holdings, cash

    for sym, is_held, cur_shares, price, w in all_targets:
        target_value = target_equity * (w / total_weight)
        target_shares = int(target_value / price) if price > 0 else 0
        if is_held:
            if target_shares < cur_shares:
                cash += (cur_shares - target_shares) * price * (1 - cost_rate)
                if target_shares > 0:
                    holdings[sym] = target_shares
                else:
                    del holdings[sym]
            elif target_shares > cur_shares:
                cost = (target_shares - cur_shares) * price * (1 + cost_rate)
                if cost <= cash:
                    cash -= cost
                    holdings[sym] = target_shares
        else:
            if target_shares > 0:
                cost = target_shares * price * (1 + cost_rate)
                if cost <= cash:
                    cash -= cost
                    holdings[sym] = target_shares
    return holdings, cash


# ============================================================
# 1. 路径配置
# ============================================================
PROJECT_ROOT = Path(r"C:/Users/haoran/Desktop/动态因子选股")
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
SELECTION_DIR = DATA_DIR / "selection"

SELECTION_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TYPES = ["insample", "outsample"]

def get_positions_file(st):
    return SELECTION_DIR / f"monthly_positions_{st}.parquet"

def get_price_file(st):
    return CLEAN_DIR / f"stock_daily_clean_{st}.parquet"

def get_benchmark_file(st):
    return DATA_DIR / "raw" / f"index_000300_raw_{st}.parquet"

def get_nav_file(st):
    return SELECTION_DIR / f"backtest_nav_{st}.parquet"

def get_performance_file(st):
    return SELECTION_DIR / f"performance_summary_{st}.csv"

def get_chart_file(st):
    return SELECTION_DIR / f"净值曲线_{st}.png"


# ============================================================
# 2. 数据加载
# ============================================================
def load_data(sample_type):
    """加载调仓指令、价格数据、基准数据"""
    print(f"[DATA] 加载 {sample_type} 数据...")
    
    positions = pd.read_parquet(get_positions_file(sample_type))
    positions['trade_date'] = pd.to_datetime(positions['trade_date'])
    print(f"   [OK] 调仓指令：{len(positions)} 条，{positions['trade_date'].nunique()} 个调仓日")
    
    price_df = pd.read_parquet(get_price_file(sample_type))
    price_df['date'] = pd.to_datetime(price_df['date'])
    print(f"   [OK] 价格数据：{len(price_df):,} 行，{price_df['symbol'].nunique()} 只股票")
    
    benchmark_df = None
    bench_file = get_benchmark_file(sample_type)
    if bench_file.exists():
        benchmark_df = pd.read_parquet(bench_file)
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
        print(f"   [OK] 基准数据：{len(benchmark_df):,} 行")
    else:
        print("   ⚠️ 未找到基准数据，将跳过超额收益计算")
    
    return positions, price_df, benchmark_df


# ============================================================
# 3. 构建价格矩阵
# ============================================================
def build_price_matrices(price_df):
    """构造收盘价矩阵和开盘价矩阵"""
    close_pivot = price_df.pivot_table(
        index='date', columns='symbol', values='close', aggfunc='last'
    ).sort_index()
    
    open_pivot = price_df.pivot_table(
        index='date', columns='symbol', values='open', aggfunc='first'
    ).sort_index()
    
    return close_pivot, open_pivot


def get_next_trade_date(date, all_dates):
    """获取 date 之后的下一个交易日"""
    pos = all_dates.get_loc(date)
    next_idx = pos + 1
    if next_idx < len(all_dates):
        return all_dates[next_idx]
    return None


def is_traded(date, sym, open_pivot):
    """判断股票当天是否正常交易（开盘价有效）"""
    if sym not in open_pivot.columns:
        return False
    price = open_pivot.loc[date, sym]
    return pd.notna(price) and price > 0


# ============================================================
# 4. 核心回测引擎
# ============================================================
def run_backtest(sample_type, initial_capital=1_000_000, cost_rate=0.001):
    """
    执行完整回测
    """
    print(f"\n[CALC] 运行 {sample_type} 回测...")
    
    # ---- 加载数据 ----
    positions, price_df, benchmark_df = load_data(sample_type)
    
    # ---- 构建矩阵 ----
    close_pivot, open_pivot = build_price_matrices(price_df)
    all_dates = close_pivot.index
    rebalance_dates = sorted(positions['trade_date'].unique())
    rebalance_dates = [d for d in rebalance_dates if d in all_dates]
    
    if len(rebalance_dates) < 2:
        raise ValueError("有效调仓日少于2个，无法回测")
    
    # ---- T+1 映射 ----
    trade_to_exec = {}
    for d in rebalance_dates:
        ed = get_next_trade_date(d, all_dates)
        if ed is not None:
            trade_to_exec[d] = ed
    if len(trade_to_exec) < 2:
        raise ValueError("调仓执行日少于2个，无法回测")
    
    exec_dates = list(trade_to_exec.values())
    first_exec = exec_dates[0]
    
    # ---- 基准价格 ----
    bench_close = None
    if benchmark_df is not None:
        bdf = benchmark_df.copy()
        bdf['date'] = pd.to_datetime(bdf['date'])
        bdf = bdf.set_index('date').sort_index()
        bench_close = bdf['index_close']
    
    # ---- 初始化账户 ----
    cash = initial_capital
    holdings = {}
    nav_records = []
    peak_nav = 1.0
    position_ratio = 1.0
    daily_returns = []
    nav_last = 1.0
    
    # ---- 遍历交易日 ----
    for date in all_dates[all_dates >= first_exec]:
        # ---------- 调仓执行 ----------
        if date in trade_to_exec.values():
            signal_date = [td for td, ed in trade_to_exec.items() if ed == date][0]
            day_positions = positions[positions['trade_date'] == signal_date]
            # ★★★ 核心改动：调用 rebalance_portfolio，传入权重 ★★★
            holdings, cash = rebalance_portfolio(
                holdings, cash, day_positions, position_ratio,
                date, open_pivot, cost_rate
            )
        
        # ---------- 每日收盘估值 ----------
        total_value = cash
        for sym, shares in holdings.items():
            if shares <= 0:
                continue
            if sym == 'CSI300' and bench_close is not None:
                price = bench_close.loc[date] if date in bench_close.index else 0
            else:
                if sym in close_pivot.columns:
                    price = close_pivot.loc[date, sym]
                    if pd.isna(price):
                        valid_prices = close_pivot[sym].dropna()
                        if not valid_prices.empty and valid_prices.index[-1] < date:
                            price = valid_prices.iloc[-1]
                        else:
                            price = 0
                else:
                    price = 0
            if price > 0:
                total_value += shares * price
        
        nav = total_value / initial_capital
        
        # ---------- 仓位管理（回撤阶梯 + 波动率目标） ----------
        if nav > peak_nav:
            peak_nav = nav
        dd = (nav - peak_nav) / peak_nav
        
        if nav_last > 0:
            daily_returns.append(nav / nav_last - 1)
        
        # ---------- 自适应仓位管理（无硬编码阈值） ----------
        if nav > peak_nav:
            peak_nav = nav
        dd = (nav - peak_nav) / peak_nav

        if nav_last > 0:
            daily_returns.append(nav / nav_last - 1)

                # 回撤阶梯（收紧）
        if dd < -0.18:
            dd_ratio = 0.40
        elif dd < -0.12:
            dd_ratio = 0.60
        elif dd < -0.07:
            dd_ratio = 0.80
        else:
            dd_ratio = 1.00

        # 波动率目标（12%）
        if len(daily_returns) >= 20:
            current_vol = np.std(daily_returns[-20:]) * np.sqrt(252)
        else:
            current_vol = 0.20
        vol_ratio = 0.12 / current_vol if current_vol > 0 else 1.0
        vol_ratio = np.clip(vol_ratio, 0.30, 1.0)

        position_ratio = min(dd_ratio, vol_ratio)

        # 记录
        nav_records.append({
            'date': date,
            'total_value': total_value,
            'cash': cash,
            'stock_value': total_value - cash,
            'position_ratio': position_ratio,
            'nav': nav
        })
        
        nav_last = nav
    
    nav_df = pd.DataFrame(nav_records)
    nav_df['nav'] = nav_df['total_value'] / initial_capital
    print(f"   [OK] 回测完成，共 {len(nav_df)} 个交易日")
    
    return nav_df, close_pivot, bench_close


# ============================================================
# 5. 绩效计算
# ============================================================
def calc_performance(nav_df, bench_close=None):
    print("[CALC] 计算绩效指标...")
    
    if len(nav_df) < 2:
        return {}
    
    total_return = nav_df['nav'].iloc[-1] - 1
    days = (nav_df['date'].iloc[-1] - nav_df['date'].iloc[0]).days
    years = days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    nav_df = nav_df.set_index('date')
    nav_df['daily_ret'] = nav_df['nav'].pct_change().fillna(0)
    annual_vol = nav_df['daily_ret'].std() * np.sqrt(252) if nav_df['daily_ret'].std() > 0 else 0
    
    risk_free = 0.02
    sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0
    
    cummax = nav_df['nav'].cummax()
    drawdown = (nav_df['nav'] - cummax) / cummax
    max_drawdown = drawdown.min()
    win_rate = (nav_df['daily_ret'] > 0).mean()
    
    excess_return = None
    information_ratio = None
    if bench_close is not None:
        bench_aligned = bench_close.reindex(nav_df.index, method='ffill')
        bench_aligned = bench_aligned / bench_aligned.iloc[0]
        if not bench_aligned.isna().all():
            bench_total_return = bench_aligned.iloc[-1] - 1
            excess_return = total_return - bench_total_return
            bench_daily_ret = bench_aligned.pct_change().fillna(0)
            excess_daily = nav_df['daily_ret'] - bench_daily_ret
            if excess_daily.std() > 0:
                excess_annual = excess_daily.mean() * 252
                excess_vol = excess_daily.std() * np.sqrt(252)
                information_ratio = excess_annual / excess_vol if excess_vol > 0 else 0
    
    result = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'days': days,
        'years': years,
        'n_days': len(nav_df),
    }
    if excess_return is not None:
        result['excess_return'] = excess_return
        result['information_ratio'] = information_ratio
    return result


# ============================================================
# 6. 绘图与输出
# ============================================================
def save_results(nav_df, perf, sample_type, bench_close=None):
    # 保存净值
    nav_file = get_nav_file(sample_type)
    perf_file = get_performance_file(sample_type)
    nav_df.to_parquet(nav_file, index=False)
    pd.DataFrame([perf]).to_csv(perf_file, index=False, encoding='utf-8-sig')
    print(f"[SAVE] 净值：{nav_file}")
    print(f"[SAVE] 绩效：{perf_file}")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(nav_df['date'], nav_df['nav'], label='策略净值', linewidth=2)
    if bench_close is not None:
        bench_norm = bench_close.reindex(pd.to_datetime(nav_df['date']), method='ffill') / bench_close.iloc[0]
        ax1.plot(nav_df['date'], bench_norm.values, label='沪深300', linewidth=1.5, linestyle='--')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_title(f'{sample_type} 净值曲线（累计收益: {perf["total_return"]*100:.2f}%）', fontsize=14)
    ax1.set_ylabel('累计净值')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    if 'position_ratio' in nav_df.columns:
        ax2.fill_between(nav_df['date'], 0, nav_df['position_ratio'], alpha=0.3, color='steelblue')
        ax2.plot(nav_df['date'], nav_df['position_ratio'], linewidth=1, color='steelblue')
        ax2.set_ylabel('仓位')
        ax2.set_ylim(0, 1.05)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(alpha=0.3)
    ax2.set_xlabel('日期')
    plt.tight_layout()
    chart_path = get_chart_file(sample_type)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 图表：{chart_path}")


# ============================================================
# 7. 主程序
# ============================================================
def main():
    print("=" * 70)
    print("📈 完整回测（样本内 / 样本外）")
    print("=" * 70)
    
    results = {}
    for st in SAMPLE_TYPES:
        try:
            nav_df, _, bench_close = run_backtest(st)
            perf = calc_performance(nav_df, bench_close)
            
            # ---- 详细报告 ----
            print("\n" + "=" * 60)
            print(f"[CALC] {st} 回测绩效报告")
            print("=" * 60)
            print(f"回测区间：{nav_df['date'].iloc[0].date()} ~ {nav_df['date'].iloc[-1].date()}")
            print(f"交易日数：{perf['n_days']} 天")
            print(f"累计收益率：{perf['total_return']*100:.2f}%")
            print(f"年化收益率：{perf['annual_return']*100:.2f}%")
            print(f"年化波动率：{perf['annual_volatility']*100:.2f}%")
            print(f"夏普比率：{perf['sharpe_ratio']:.4f}")
            print(f"最大回撤：{perf['max_drawdown']*100:.2f}%")
            print(f"胜率：{perf['win_rate']*100:.2f}%")
            if 'position_ratio' in nav_df.columns:
                print(f"平均仓位：{nav_df['position_ratio'].mean()*100:.1f}%"
                      f"（最低: {nav_df['position_ratio'].min()*100:.1f}%）")
            if perf.get('excess_return') is not None:
                print(f"超额收益（vs 沪深300）：{perf['excess_return']*100:.2f}%")
                if perf.get('information_ratio') is not None:
                    print(f"信息比率：{perf['information_ratio']:.4f}")
            
            # ---- 分年收益 ----
            print("\n" + "-" * 70)
            print(f">> 分年收益（vs 沪深300）")
            print("-" * 70)
            nav_df['year'] = pd.to_datetime(nav_df['date']).dt.year
            yearly = nav_df.groupby('year').agg(
                start_nav=('nav', 'first'),
                end_nav=('nav', 'last'),
                max_dd=('nav', lambda x: ((x / x.cummax()) - 1).min()),
            )
            yearly['策略收益'] = yearly['end_nav'] / yearly['start_nav'] - 1
            yearly['策略累计净值'] = (1 + yearly['策略收益']).cumprod()
            
            if bench_close is not None:
                bench_yearly = bench_close.groupby(bench_close.index.year).agg(first='first', last='last')
                bench_yearly['沪深300收益'] = bench_yearly['last'] / bench_yearly['first'] - 1
                bench_yearly.index = bench_yearly.index.astype(int)
                yearly = yearly.join(bench_yearly['沪深300收益'])
                yearly['超额收益'] = yearly['策略收益'] - yearly['沪深300收益']
            
            header = f"{'年份':<8}{'策略收益':>10}{'沪深300收益':>12}{'超额收益':>10}{'策略累计净值':>14}{'策略最大回撤':>12}"
            print(header)
            for yr, row in yearly.iterrows():
                bench_str = f"{row['沪深300收益']*100:>11.2f}%" if '沪深300收益' in yearly.columns and pd.notna(row.get('沪深300收益')) else f"{'N/A':>12}"
                excess_str = f"{row['超额收益']*100:>9.2f}%" if '超额收益' in yearly.columns and pd.notna(row.get('超额收益')) else f"{'N/A':>10}"
                print(f"{yr:<8}{row['策略收益']*100:>9.2f}%{bench_str}{excess_str}{row['策略累计净值']:>13.4f}{row['max_dd']*100:>11.2f}%")
            
            save_results(nav_df, perf, st, bench_close)
            results[st] = {'nav': nav_df, 'perf': perf}
        except FileNotFoundError as e:
            print(f"⚠️ [{st}] 跳过：{e}")
        except Exception as e:
            import traceback
            print(f"❌ [{st}] 失败：{e}")
            traceback.print_exc()
    
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("📊 样本内外对比")
        print("=" * 70)
        comp = pd.DataFrame({
            st: results[st]['perf'] for st in results
        }).T
        cols = ['total_return', 'annual_return', 'annual_volatility',
                'sharpe_ratio', 'max_drawdown', 'win_rate',
                'excess_return', 'information_ratio']
        available = [c for c in cols if c in comp.columns]
        print(comp[available].to_string())
        comp.to_csv(SELECTION_DIR / "performance_comparison.csv", encoding='utf-8-sig')
        print(f"\n[OK] 对比报告已保存至：{SELECTION_DIR / 'performance_comparison.csv'}")
    
    print("\n[OK] 全部完成！")


if __name__ == "__main__":
    main()
