# -*- coding: utf-8 -*-
"""
可视化模块 —— 策略报告图表批量生成
==================================================
功能：
  - 净值 + 回撤曲线
  - 年度收益柱状图（含基准对比）
  - 滚动60日夏普比率
  - 月度收益率热力图
  - 仓位变化图
  - 样本内外雷达对比
  - 因子数量月度柱状图

输入：data/selection/backtest_nav_*.parquet
      data/selection/performance_summary_*.csv
      data/factors/rolling_selection_log_*.csv
输出：data/charts/ 下各类 PNG 图表
==================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 路径配置
# ============================================================
PROJECT_ROOT = Path(r"C:/Users/haoran/Desktop/动态因子选股")
DATA_DIR     = PROJECT_ROOT / "data"
SELECTION_DIR = DATA_DIR / "selection"
FACTOR_DIR   = DATA_DIR / "factors"
CHART_DIR    = DATA_DIR / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TYPES = ["insample", "outsample"]


# ============================================================
# 1. 数据加载
# ============================================================
def load_backtest_results():
    """
    从已保存的回测结果文件中加载净值+绩效，组装为 results dict。
    返回：results = { 'insample': {'nav': df, 'perf': dict}, 'outsample': {...} }
    """
    results = {}
    for st in SAMPLE_TYPES:
        nav_file = SELECTION_DIR / f"backtest_nav_{st}.parquet"
        perf_file = SELECTION_DIR / f"performance_summary_{st}.csv"
        if not nav_file.exists():
            print(f"   ⚠️ [{st}] 未找到净值文件：{nav_file}，跳过")
            continue
        nav_df = pd.read_parquet(nav_file)
        perf = {}
        if perf_file.exists():
            perf_df = pd.read_csv(perf_file)
            perf = perf_df.iloc[0].to_dict() if len(perf_df) > 0 else {}
        results[st] = {'nav': nav_df, 'perf': perf}
        print(f"   [OK] {st}: {len(nav_df)} 个交易日, perf keys={list(perf.keys())[:6]}...")
    return results


# ============================================================
# 2. 图表生成
# ============================================================
def generate_report_charts(results, output_dir):
    """
    批量生成策略报告图表
    results : dict, 键为 sample_type, 值为 {'nav': nav_df, 'perf': perf, ...}
    output_dir : Path, 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n[CHART] 生成报告图表...")

    # ---------- 1. 净值 + 回撤曲线（每个样本） ----------
    for st, data in results.items():
        nav_df = data['nav'].copy()
        if nav_df is None or len(nav_df) == 0:
            continue
        nav_df['date'] = pd.to_datetime(nav_df['date'])
        nav_df = nav_df.sort_values('date')
        nav_df['drawdown'] = nav_df['nav'] / nav_df['nav'].cummax() - 1

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})
        # 净值
        ax1.plot(nav_df['date'], nav_df['nav'], label='策略净值', linewidth=2)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f'{st} 净值与回撤')
        ax1.legend()
        ax1.grid(alpha=0.3)
        # 回撤
        ax2.fill_between(nav_df['date'], 0, nav_df['drawdown'], color='red', alpha=0.3)
        ax2.plot(nav_df['date'], nav_df['drawdown'], color='darkred', linewidth=1)
        ax2.axhline(y=-0.12, color='blue', linestyle='--', linewidth=1, label='-12% 阈值')
        ax2.axhline(y=-0.05, color='green', linestyle='--', linewidth=1, label='-5% 阈值')
        ax2.set_ylabel('回撤')
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{st}_nav_drawdown.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ---------- 2. 年度收益对比柱状图（每个样本） ----------
    for st, data in results.items():
        nav_df = data['nav'].copy()
        if nav_df is None or len(nav_df) < 2:
            continue
        nav_df['date'] = pd.to_datetime(nav_df['date'])
        nav_df['year'] = nav_df['date'].dt.year
        # 计算年度收益
        yearly = nav_df.groupby('year').agg(start=('nav', 'first'), end=('nav', 'last'))
        yearly['strategy'] = yearly['end'] / yearly['start'] - 1
        # 尝试加载基准收益（若存在）
        bench_yearly = None
        # 如果有 perf 中包含 excess_return 且知道年份，但这里简单处理：直接从 nav_df 计算基准（若 data 中有 bench_close 字段）
        # 由于 data 中可能没有 bench_close，我们尽量从 perf 或外部获取；若没有，只画策略收益
        fig, ax = plt.subplots(figsize=(10, 6))
        years = yearly.index.astype(str)
        ax.bar(years, yearly['strategy'] * 100, label='策略收益', color='steelblue', alpha=0.7)
        # 若有超额收益信息，可叠加标注
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{st} 年度收益')
        ax.set_ylabel('收益率 (%)')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{st}_yearly_returns.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ---------- 3. 滚动夏普（60日） ----------
    for st, data in results.items():
        nav_df = data['nav'].copy()
        if nav_df is None or len(nav_df) < 60:
            continue
        nav_df['date'] = pd.to_datetime(nav_df['date'])
        nav_df = nav_df.sort_values('date')
        nav_df['daily_ret'] = nav_df['nav'].pct_change().fillna(0)
        # 滚动60日夏普（年化）
        rolling_mean = nav_df['daily_ret'].rolling(60).mean() * 252
        rolling_std = nav_df['daily_ret'].rolling(60).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - 0.02) / rolling_std
        rolling_sharpe = rolling_sharpe.fillna(0)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(nav_df['date'], rolling_sharpe, color='purple', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{st} 滚动60日夏普比率')
        ax.set_ylabel('夏普')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{st}_rolling_sharpe.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ---------- 4. 月度收益率热力图（需 seaborn） ----------
    try:
        import seaborn as sns
        for st, data in results.items():
            nav_df = data['nav'].copy()
            if nav_df is None or len(nav_df) < 30:
                continue
            nav_df['date'] = pd.to_datetime(nav_df['date'])
            nav_df['year'] = nav_df['date'].dt.year
            nav_df['month'] = nav_df['date'].dt.month
            nav_df['daily_ret'] = nav_df['nav'].pct_change().fillna(0)
            # 月度累计收益
            monthly = nav_df.groupby(['year', 'month'])['daily_ret'].apply(lambda x: (1 + x).prod() - 1).reset_index()
            pivot = monthly.pivot(index='year', columns='month', values='daily_ret')
            # 填充缺失月份为0
            pivot = pivot.reindex(columns=range(1,13), fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': '月度收益'})
            ax.set_title(f'{st} 月度收益热力图')
            plt.tight_layout()
            plt.savefig(output_dir / f'{st}_monthly_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
    except ImportError:
        print("   [WARN] seaborn未安装，跳过月度热力图")

    # ---------- 5. 仓位比例变化（若存在） ----------
    for st, data in results.items():
        nav_df = data['nav'].copy()
        if nav_df is None or 'position_ratio' not in nav_df.columns:
            continue
        nav_df['date'] = pd.to_datetime(nav_df['date'])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(nav_df['date'], 0, nav_df['position_ratio'], alpha=0.3, color='steelblue')
        ax.plot(nav_df['date'], nav_df['position_ratio'], color='steelblue', linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('仓位比例')
        ax.set_title(f'{st} 仓位变化')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{st}_position_ratio.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ---------- 6. 样本内外对比雷达图 ----------
    if 'insample' in results and 'outsample' in results:
        try:
            metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            metric_labels = ['累计收益', '年化收益', '夏普比率', '最大回撤', '胜率']
            # 提取并归一化
            df_comp = pd.DataFrame({
                st: {m: results[st]['perf'].get(m, 0) for m in metrics}
                for st in ['insample', 'outsample']
            }).T
            # 对每个指标做 Min-Max 归一化（除最大回撤取负值）
            df_norm = df_comp.copy()
            for m in metrics:
                if m == 'max_drawdown':
                    # 回撤取相反数，越大越好
                    df_norm[m] = -df_norm[m]
                # Min-Max
                min_val = df_norm[m].min()
                max_val = df_norm[m].max()
                if max_val - min_val != 0:
                    df_norm[m] = (df_norm[m] - min_val) / (max_val - min_val)
                else:
                    df_norm[m] = 0.5
            # 绘制雷达图
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            for st in ['insample', 'outsample']:
                values = df_norm.loc[st].values.flatten().tolist()
                values += values[:1]
                ax.plot(angles, values, label=st, linewidth=2)
                ax.fill(angles, values, alpha=0.1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.legend(loc='upper right')
            ax.set_title('样本内 vs 样本外 雷达对比', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / 'insample_outsample_radar.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"   [WARN] 雷达图生成失败：{e}")

    # ---------- 7. 因子数量月度柱状图 ----------
    log_path = FACTOR_DIR / "rolling_selection_log_insample.csv"
    if log_path.exists():
        try:
            log_df = pd.read_csv(log_path)
            log_df.columns = log_df.columns.str.strip().str.lower()
            if 'year_month' in log_df.columns and 'factors' in log_df.columns:
                # 解析因子家族
                # 由于因子列表是逗号分隔字符串，统计出现频率（可简化）
                # 这里只展示每月因子数量
                log_df['year_month'] = pd.to_datetime(log_df['year_month'])
                log_df = log_df.sort_values('year_month')
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.bar(log_df['year_month'].astype(str), log_df['n_factors'], color='teal', alpha=0.6)
                ax.set_title('每月有效因子数量')
                ax.set_ylabel('因子数量')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_dir / 'factor_count_monthly.png', dpi=150, bbox_inches='tight')
                plt.close()
        except:
            pass

    print(f"   [OK] 图表已保存至：{output_dir}")


# ============================================================
# 3. 主入口
# ============================================================
def main():
    print("=" * 70)
    print("📊 策略可视化 — 批量生成报告图表")
    print("=" * 70)

    # 加载回测结果
    print("\n[DATA] 加载回测结果...")
    results = load_backtest_results()
    if not results:
        print("⚠️ 未找到任何回测结果，请先运行 回测.py")
        return

    # 生成图表
    generate_report_charts(results, CHART_DIR)

    print("\n[OK] 可视化全部完成！")


if __name__ == "__main__":
    main()