# ============================================================
# 🧬 因子计算 —— FactorCalculator 类
# ============================================================
# 功能：
#   1. 技术因子：动量 × N期、波动率 × N期、换手率
#   2. 估值因子：PE_TTM、PB、PS_TTM、总市值
#   3. 质量+成长因子：ROE、净利润同比、毛利率、ROE同比变化、资产负债率
#   4. 情绪因子：成交量变化率、量比
#   5. 缺失值填充 + 缩尾处理 + Z-score 截面标准化
# ============================================================
# 时间范围：2021-01-01 ~ 2026-06-30
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
FACTOR_DIR = DATA_DIR / "factors"
FACTOR_DIR.mkdir(parents=True, exist_ok=True)

# ---- 输入文件 ----
INPUT_MERGED = CLEAN_DIR / "stock_merged_clean.parquet"

# ---- 输出文件（样本内 insample: 20180101-20231231） ----
OUTPUT_IN_PARQUET = FACTOR_DIR / "all_factors_insample.parquet"
OUTPUT_IN_CSV     = FACTOR_DIR / "all_factors_insample.csv"

# ---- 输出文件（样本外 outsample: 20240101-20260630） ----
OUTPUT_OUT_PARQUET = FACTOR_DIR / "all_factors_outsample.parquet"
OUTPUT_OUT_CSV     = FACTOR_DIR / "all_factors_outsample.csv"

# ---- 拆分日期 ----
TRAIN_START = "2018-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2026-06-30"


class FactorCalculator:
    """
    多因子计算器：统一计算候选因子（技术 + 估值 + 财务 + 情绪）

    Parameters:
        technical_windows: list of int, 技术指标计算窗口（默认 [5, 10, 20, 60, 120, 240]）
        winsorize_quantile: float, 缩尾处理的百分位（默认 0.99）
        fillna_method: str, 缺失值填充方法（'mean', 'median', 'ffill'）
        apply_standardize: bool, 是否对因子做截面 Z-score 标准化（默认 True）
        include_valuation: bool, 是否计算估值因子
        include_financial: bool, 是否计算财务因子
        include_sentiment: bool, 是否计算情绪因子
    """

    def __init__(self, technical_windows=None, winsorize_quantile=0.99,
                 fillna_method='mean', apply_standardize=True,
                 include_valuation=True, include_financial=True,
                 include_sentiment=True):
        self.technical_windows = technical_windows or [5, 10, 20, 60, 120, 240]
        self.winsorize_quantile = winsorize_quantile
        self.fillna_method = fillna_method
        self.apply_standardize = apply_standardize
        self.include_valuation = include_valuation
        self.include_financial = include_financial
        self.include_sentiment = include_sentiment

        # 因子分类存储
        self.technical_factor_cols = []
        self.valuation_factor_cols = []
        self.financial_factor_cols = []
        self.growth_factor_cols = []
        self.sentiment_factor_cols = []

    # ============================================================
    # 1. 技术因子：多周期动量
    # ============================================================

    def _compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """多周期动量因子（包含 240 日年线动量）"""
        for window in self.technical_windows:
            col = f'momentum_{window}'
            df[col] = df.groupby('symbol')['close'].pct_change(window)
            self.technical_factor_cols.append(col)
        return df

    # ============================================================
    # 2. 技术因子：多周期波动率（年化）
    # ============================================================

    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """多周期波动率因子（年化）"""
        for window in self.technical_windows:
            col = f'volatility_{window}'
            df[col] = df.groupby('symbol')['daily_return'].transform(
                lambda x: x.rolling(window).std() * np.sqrt(252)
            )
            self.technical_factor_cols.append(col)
        return df

    # ============================================================
    # 3. 技术因子：换手率
    # ============================================================

    def _compute_turnover(self, df: pd.DataFrame) -> pd.DataFrame:
        """换手率因子（需要流通股本）"""
        # Tushare 日线成交量字段为 vol
        vol_col = 'vol' if 'vol' in df.columns else ('volume' if 'volume' in df.columns else None)
        if 'float_share' in df.columns and vol_col:
            df['turnover_rate'] = df[vol_col] / df['float_share']
            df['avg_turnover_20'] = df.groupby('symbol')['turnover_rate'].transform(
                lambda x: x.rolling(20).mean()
            )
            self.technical_factor_cols.extend(['turnover_rate', 'avg_turnover_20'])
        else:
            print("⚠️ 缺少流通股本（float_share）或成交量，跳过换手率因子")
        return df

    # ============================================================
    # 4. 技术因子：多周期最大回撤
    # ============================================================

    def _compute_max_drawdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        多周期最大回撤因子（过去N日最大跌幅）
        数值为负（如 -0.25 代表跌了 25%），合成时取负号（越低越好）
        """
        for window in self.technical_windows:  # [5, 10, 20, 60, 120, 240]
            col = f'max_drawdown_{window}'
            # 计算回撤：当前价 / 窗口内最高价 - 1，再取窗口内最小值
            df[col] = df.groupby('symbol')['close'].transform(
                lambda x: (x / x.rolling(window, min_periods=1).max() - 1)
                           .rolling(window, min_periods=1).min()
            )
            self.technical_factor_cols.append(col)
        return df

    # ============================================================
    # 5. 估值因子（低估值负向）
    # ============================================================

    def _compute_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """估值因子：PE, PB, PS, 市值（市值单独用）"""
        if not self.include_valuation:
            return df

        for col in ['pe_ttm', 'pb', 'ps_ttm']:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                self.valuation_factor_cols.append(col)
            else:
                print(f"⚠️ 缺少 {col}，跳过该估值因子")

        # 总市值（小市值正向，单独处理，放在估值组便于后续提取）
        if 'total_mv' in df.columns:
            df['total_mv'] = df['total_mv'].replace([np.inf, -np.inf], np.nan)
            self.valuation_factor_cols.append('total_mv')

        return df

    # ============================================================
    # 6. 质量 + 成长因子（正向）
    # ============================================================

    def _compute_financial(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        质量与成长因子：
        - 质量：ROE（净资产收益率），毛利率（grossprofit_margin）
        - 成长：净利润增速（netprofit_yoy），营收增速（revenue_yoy）
        - 新增：ROE同比变化（roe_yoy_change），捕捉业绩加速
        """
        if not self.include_financial:
            return df

        # ---- 质量因子 ----
        if 'roe' in df.columns:
            df['roe'] = df['roe'].replace([np.inf, -np.inf], np.nan)
            self.financial_factor_cols.append('roe')

        if 'grossprofit_margin' in df.columns:
            df['grossprofit_margin'] = df['grossprofit_margin'].replace([np.inf, -np.inf], np.nan)
            self.financial_factor_cols.append('grossprofit_margin')

        # ---- 成长因子（直接使用Tushare提供的同比增速） ----
        if 'netprofit_yoy' in df.columns:
            df['netprofit_yoy'] = df['netprofit_yoy'].replace([np.inf, -np.inf], np.nan)
            self.financial_factor_cols.append('netprofit_yoy')
            self.growth_factor_cols.append('netprofit_yoy')   # 标记为成长

        if 'revenue_yoy' in df.columns:
            df['revenue_yoy'] = df['revenue_yoy'].replace([np.inf, -np.inf], np.nan)
            self.financial_factor_cols.append('revenue_yoy')
            self.growth_factor_cols.append('revenue_yoy')

        # ---- 新增：ROE同比变化（加速度） ----
        if 'roe' in df.columns:
            # 按股票分组计算ROE的同比变化（季度数据，shift(4)代表去年同期）
            df['roe_yoy_change'] = df.groupby('symbol')['roe'].transform(
                lambda x: x - x.shift(4)   # 今年ROE - 去年同季度ROE
            )
            df['roe_yoy_change'] = df['roe_yoy_change'].replace([np.inf, -np.inf], np.nan)
            self.financial_factor_cols.append('roe_yoy_change')
            self.growth_factor_cols.append('roe_yoy_change')
            print("✅ 新增 ROE 同比变化因子（roe_yoy_change）")

        # ---- 资产负债率（财务健康，负向） ----
        if 'debt_to_assets' in df.columns:
            df['debt_to_assets'] = df['debt_to_assets'].replace([np.inf, -np.inf], np.nan)
            self.financial_factor_cols.append('debt_to_assets')

        return df

    # ============================================================
    # 7. 情绪因子（量价）
    # ============================================================

    def _compute_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """情绪因子：成交量变化、量比等"""
        if not self.include_sentiment:
            return df

        if 'vol' in df.columns:
            # 成交量变化率（5日均量 / 20日均量 - 1）
            df['vol_ma_5'] = df.groupby('symbol')['vol'].transform(lambda x: x.rolling(5).mean())
            df['vol_ma_20'] = df.groupby('symbol')['vol'].transform(lambda x: x.rolling(20).mean())
            df['volume_change_ratio'] = df['vol_ma_5'] / df['vol_ma_20'] - 1
            self.sentiment_factor_cols.append('volume_change_ratio')
            df.drop(columns=['vol_ma_5', 'vol_ma_20'], inplace=True)

            # 量比（当日成交量 / 过去5日均量）
            df['vol_ma_5'] = df.groupby('symbol')['vol'].transform(lambda x: x.rolling(5).mean())
            df['volume_ratio'] = df['vol'] / df['vol_ma_5']
            self.sentiment_factor_cols.append('volume_ratio')
            df.drop(columns=['vol_ma_5'], inplace=True)
        else:
            print("⚠️ 缺少成交量（vol），跳过情绪因子")
        return df

    # ============================================================
    # 8. 辅助方法：缩尾处理 & 缺失值填充
    # ============================================================

    def _winsorize(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """缩尾处理：截断极端值"""
        if self.winsorize_quantile is None or self.winsorize_quantile >= 1.0:
            return df
        lower_q = 1 - self.winsorize_quantile
        upper_q = self.winsorize_quantile
        for col in cols:
            if col not in df.columns:
                continue
            def _clip_group(g):
                lower = g.quantile(lower_q)
                upper = g.quantile(upper_q)
                return g.clip(lower, upper)
            df[col] = df.groupby('date')[col].transform(_clip_group)
        return df

    def _fill_missing(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """填充缺失值"""
        if self.fillna_method == 'mean':
            for col in cols:
                if col in df.columns:
                    df[col] = df.groupby('date')[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
        elif self.fillna_method == 'median':
            for col in cols:
                if col in df.columns:
                    df[col] = df.groupby('date')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
        elif self.fillna_method == 'ffill':
            df[cols] = df.groupby('symbol')[cols].transform(lambda x: x.ffill())
        return df

    def _standardize(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        标准化（Z-score）：对每个因子列按截面（同一日期）做标准化。

        处理步骤：
          1. 去均值（中心化）
          2. 除以标准差
          3. 对标准差为 0 的列跳过，避免除零错误

        Parameters:
            cols: 需要进行标准化的因子列名列表
        """
        for col in cols:
            if col not in df.columns:
                continue
            def _zscore(g):
                mean = g.mean()
                std = g.std()
                if std == 0 or pd.isna(std):
                    return g * 0  # 标准差为 0 时全部置 0
                return (g - mean) / std
            df[col] = df.groupby('date')[col].transform(_zscore)
        return df

    # ============================================================
    # 9. 工具方法
    # ============================================================

    def get_factor_names(self):
        """返回各分类因子名称列表"""
        return {
            'technical': list(self.technical_factor_cols),
            'valuation': list(self.valuation_factor_cols),
            'financial': list(self.financial_factor_cols),
            'growth': list(self.growth_factor_cols),
            'sentiment': list(self.sentiment_factor_cols),
            'all': (self.technical_factor_cols + self.valuation_factor_cols +
                    self.financial_factor_cols + self.sentiment_factor_cols)
        }

    # ============================================================
    # 10. 主方法：计算全部因子
    # ============================================================

    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算全部因子"""
        result = df.copy()

        print("📊 计算动量因子...")
        result = self._compute_momentum(result)

        print("📊 计算波动率因子...")
        result = self._compute_volatility(result)

        print("📊 计算换手率因子...")
        result = self._compute_turnover(result)

        print("📊 计算最大回撤因子...")
        result = self._compute_max_drawdown(result)

        if self.include_valuation:
            print("📊 计算估值因子...")
            result = self._compute_valuation(result)

        if self.include_financial:
            print("📊 计算财务+成长因子...")
            result = self._compute_financial(result)

        if self.include_sentiment:
            print("📊 计算情绪因子...")
            result = self._compute_sentiment(result)

        # 合并所有因子列
        all_factor_cols = (self.technical_factor_cols + self.valuation_factor_cols +
                           self.financial_factor_cols + self.sentiment_factor_cols)

        # 填充缺失值 + 缩尾处理
        print("📊 填充缺失值 + 缩尾处理...")
        result = self._fill_missing(result, all_factor_cols)
        result = self._winsorize(result, all_factor_cols)

        # 标准化（Z-score，按截面对每个因子做标准化）
        if self.apply_standardize:
            print("📊 截面 Z-score 标准化...")
            result = self._standardize(result, all_factor_cols)

        print(f"\n✅ 因子计算完成！共生成 {len(all_factor_cols)} 个因子")
        print(f"  技术因子：{len(self.technical_factor_cols)} 个 — {self.technical_factor_cols}")
        print(f"  估值因子：{len(self.valuation_factor_cols)} 个 — {self.valuation_factor_cols}")
        print(f"  财务因子：{len(self.financial_factor_cols)} 个 — {self.financial_factor_cols}")
        print(f"  情绪因子：{len(self.sentiment_factor_cols)} 个 — {self.sentiment_factor_cols}")

        return result


# ============================================================
# 样本内外拆分 + 分别计算因子
# ============================================================

def split_and_compute_factors(
    merged_path=INPUT_MERGED,
    train_start=TRAIN_START, train_end=TRAIN_END,
    test_start=TEST_START,   test_end=TEST_END,
    calculator_kwargs=None,
):
    """
    加载合并对齐数据，按日期拆分为样本内/外，分别计算因子并保存。

    Parameters:
        merged_path:       合并对齐数据文件路径
        train_start/end:   样本内日期范围
        test_start/end:    样本外日期范围
        calculator_kwargs: FactorCalculator 构造参数 dict

    Returns:
        dict: {
            'insample':  (factor_df_insample, FactorCalculator),
            'outsample': (factor_df_outsample, FactorCalculator),
        }
    """
    if calculator_kwargs is None:
        calculator_kwargs = {}

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts   = pd.Timestamp(train_end)
    test_start_ts  = pd.Timestamp(test_start)
    test_end_ts    = pd.Timestamp(test_end)

    # ---- 加载数据 ----
    print("=" * 60)
    print("📁 加载合并对齐数据")
    print("=" * 60)
    print(f"  文件：{merged_path}")
    merged = pd.read_parquet(merged_path)
    print(f"  全量：{len(merged):,} 行 × {len(merged.columns)} 列")
    if 'date' in merged.columns:
        merged['date'] = pd.to_datetime(merged['date'])
        print(f"  日期范围：{merged['date'].min().date()} ~ {merged['date'].max().date()}")

    # ---- 拆分 ----
    print("\n" + "=" * 60)
    print("✂️ 样本内外拆分")
    print("=" * 60)
    print(f"  样本内 train: {train_start} ~ {train_end}")
    print(f"  样本外 test:  {test_start} ~ {test_end}")

    insample = merged[(merged['date'] >= train_start_ts) & (merged['date'] <= train_end_ts)].copy()
    outsample = merged[(merged['date'] >= test_start_ts) & (merged['date'] <= test_end_ts)].copy()

    print(f"  样本内：{len(insample):,} 行  ({insample['date'].min().date()} ~ {insample['date'].max().date()})")
    print(f"  样本外：{len(outsample):,} 行  ({outsample['date'].min().date()} ~ {outsample['date'].max().date()})")

    # ---- 分别计算因子 ----
    results = {}

    for label, df, out_pq, out_csv in [
        ('样本内 insample', insample,
         OUTPUT_IN_PARQUET, OUTPUT_IN_CSV),
        ('样本外 outsample', outsample,
         OUTPUT_OUT_PARQUET, OUTPUT_OUT_CSV),
    ]:
        print(f"\n{'=' * 60}")
        print(f"🧬 计算因子 — {label}")
        print(f"{'=' * 60}")

        calc = FactorCalculator(**calculator_kwargs)
        factor_df = calc.compute_all_factors(df)

        # 保存
        factor_df.to_parquet(out_pq)
        factor_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"💾 {label}因子已保存：{out_pq}")
        print(f"💾 {label}因子已保存：{out_csv}")

        results[label] = (factor_df, calc)

    return results


# ============================================================
# 主执行入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧬 因子计算模块 - 沪深300多因子选股")
    print("=" * 60)

    # 公共参数
    calc_kwargs = dict(
        technical_windows=[5, 10, 20, 60, 120, 240],
        winsorize_quantile=0.99,
        fillna_method='mean',
        apply_standardize=True,
        include_valuation=True,
        include_financial=True,
        include_sentiment=True,
    )

    # ---- 样本内外拆分后分别计算因子 ----
    split_results = split_and_compute_factors(
        merged_path=INPUT_MERGED,
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START,   test_end=TEST_END,
        calculator_kwargs=calc_kwargs,
    )

    print(f"\n{'=' * 60}")
    print(f"✅ 全部完成！")
    print(f"{'=' * 60}")
    print(f"  样本内因子：{OUTPUT_IN_PARQUET}")
    print(f"  样本外因子：{OUTPUT_OUT_PARQUET}")
