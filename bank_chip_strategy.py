import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 参数配置 ====================
FILE_BANK = "每日基金净值与行情_515020.SH.xls"
FILE_CHIP = "每日基金净值与行情_588200.SH.xls"
SHEET_NAME = "每日净值"
BANK_WEIGHT = 0.6
CHIP_WEIGHT = 1 - BANK_WEIGHT
REBALANCE_FREQ = 'Q'          # 'M' 月度调仓，'Q' 季度调仓
RISK_FREE_RATE = 0.025

# 保存文件名配置
SAVE_NAV_FILE = "portfolio_nav.csv"
SAVE_METRICS_FILE = "metrics.csv"
SAVE_FIGURE_FILE = "nav_curve.png"

# ==================== 1. 读取数据 ====================
def load_price(file_path, price_col='收盘价(元)'):
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME, skipfooter=1)
    df['日期'] = pd.to_datetime(df['日期'])
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    df = df.drop_duplicates(subset='日期', keep='last')
    df = df.dropna(subset=[price_col])
    df = df.set_index('日期')[price_col].to_frame()
    code = file_path.split('_')[1].split('.')[0]
    df.columns = [code]
    return df

print("正在读取银行ETF数据...")
df_bank = load_price(FILE_BANK)
print("正在读取芯片ETF数据...")
df_chip = load_price(FILE_CHIP)

assert df_bank.index.is_unique, "银行数据索引有重复"
assert df_chip.index.is_unique, "芯片数据索引有重复"

df = pd.concat([df_bank, df_chip], axis=1).sort_index()
df.ffill(inplace=True)
df.dropna(inplace=True)
df.columns = ['bank', 'chip']

print("数据合并完成")
print(f"时间范围：{df.index.min().date()} 至 {df.index.max().date()}")
print(f"有效交易日数量：{len(df)}")
print("\n前5行数据：")
print(df.head())

# ==================== 2. 策略回测（稳健获取每周期最后一个交易日） ====================
# 使用 groupby + apply 直接提取每个周期（月/季）的最后一个交易日索引
if REBALANCE_FREQ == 'M':
    rebalance_dates = df.groupby(df.index.to_period('M')).apply(lambda x: x.index[-1])
elif REBALANCE_FREQ == 'Q':
    rebalance_dates = df.groupby(df.index.to_period('Q')).apply(lambda x: x.index[-1])
else:
    raise ValueError("REBALANCE_FREQ 仅支持 'M' 或 'Q'")

# 确保 rebalance_dates 是 DatetimeIndex，且去除可能的多余索引级别
if isinstance(rebalance_dates, pd.Series):
    rebalance_dates = pd.DatetimeIndex(rebalance_dates.values)

df['is_rebalance'] = df.index.isin(rebalance_dates)

# 初始化组合净值序列和权重记录
portfolio_nav = pd.Series(1.0, index=df.index)
weights = pd.DataFrame(index=df.index, columns=['bank_w', 'chip_w'])

first_date = df.index[0]
bank_price0 = df.loc[first_date, 'bank']
chip_price0 = df.loc[first_date, 'chip']
capital0 = 1.0
bank_shares = (BANK_WEIGHT * capital0) / bank_price0
chip_shares = (CHIP_WEIGHT * capital0) / chip_price0

for i, date in enumerate(df.index):
    bank_price = df.loc[date, 'bank']
    chip_price = df.loc[date, 'chip']
    
    if i == 0:
        nav = bank_shares * bank_price + chip_shares * chip_price
        portfolio_nav.loc[date] = nav
        weights.loc[date, 'bank_w'] = (bank_shares * bank_price) / nav
        weights.loc[date, 'chip_w'] = (chip_shares * chip_price) / nav
    else:
        curr_bank_val = bank_shares * bank_price
        curr_chip_val = chip_shares * chip_price
        curr_total = curr_bank_val + curr_chip_val
        portfolio_nav.loc[date] = curr_total
        
        weights.loc[date, 'bank_w'] = curr_bank_val / curr_total
        weights.loc[date, 'chip_w'] = curr_chip_val / curr_total
        
        if df.loc[date, 'is_rebalance']:
            target_bank_val = BANK_WEIGHT * curr_total
            target_chip_val = CHIP_WEIGHT * curr_total
            bank_shares = target_bank_val / bank_price
            chip_shares = target_chip_val / chip_price
            weights.loc[date, 'bank_w'] = BANK_WEIGHT
            weights.loc[date, 'chip_w'] = CHIP_WEIGHT

# ==================== 3. 计算绩效指标 ====================
def calculate_metrics(nav_series, rf_annual=RISK_FREE_RATE, periods_per_year=252):
    ret = nav_series.pct_change().dropna()
    total_days = len(ret)
    years = total_days / periods_per_year
    
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = (annual_return - rf_annual) / annual_vol if annual_vol != 0 else 0
    
    cummax = nav_series.expanding().max()
    drawdown = (nav_series - cummax) / cummax
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        '总收益率': total_return,
        '年化收益率': annual_return,
        '年化波动率': annual_vol,
        '夏普比率': sharpe,
        '最大回撤': max_drawdown,
        'Calmar比率': calmar,
    }

metrics = calculate_metrics(portfolio_nav)

print("\n" + "="*40)
print("策略绩效指标")
print("="*40)
for key, val in metrics.items():
    if key in ['总收益率', '年化收益率', '年化波动率', '最大回撤']:
        print(f"{key:10s}: {val:>8.2%}")
    else:
        print(f"{key:10s}: {val:>8.4f}")

# ==================== 4. 保存净值、指标、图片 ====================
portfolio_nav.to_csv(SAVE_NAV_FILE, header=True, index_label='日期')
print(f"\n净值序列已保存至：{os.path.abspath(SAVE_NAV_FILE)}")

metrics_df = pd.DataFrame(list(metrics.items()), columns=['指标', '数值'])
metrics_df.to_csv(SAVE_METRICS_FILE, index=False, encoding='utf-8-sig')
print(f"绩效指标已保存至：{os.path.abspath(SAVE_METRICS_FILE)}")

# ==================== 5. 绘制净值曲线并保存图片 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 6))

# 策略净值曲线
plt.plot(portfolio_nav.index, portfolio_nav, 
         label=f'平衡策略 ({BANK_WEIGHT:.0%}银行 / {CHIP_WEIGHT:.0%}芯片)', 
         linewidth=1.8, color='blue')

# 单个ETF归一化曲线（对比用）
bank_norm = df['bank'] / df['bank'].iloc[0]
chip_norm = df['chip'] / df['chip'].iloc[0]
plt.plot(df.index, bank_norm, '--', alpha=0.6, label='银行ETF (归一化)', color='green')
plt.plot(df.index, chip_norm, '--', alpha=0.6, label='芯片ETF (归一化)', color='red')

# 标记调仓位置（红色三角形）
rebalance_idx = df[df['is_rebalance']].index
rebalance_nav = portfolio_nav[rebalance_idx]
plt.scatter(rebalance_idx, rebalance_nav, marker='^', s=80, c='red',
            label='调仓点', zorder=5)

freq_name = '月度' if REBALANCE_FREQ == 'M' else '季度'
plt.title(f'{freq_name}再平衡策略净值曲线 (再平衡频率: {REBALANCE_FREQ})')
plt.xlabel('日期')
plt.ylabel('累计净值')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(SAVE_FIGURE_FILE, dpi=300, bbox_inches='tight')
print(f"净值曲线图已保存至：{os.path.abspath(SAVE_FIGURE_FILE)}")

plt.show()

# ==================== 6. 输出再平衡日期 ====================
print(f"\n再平衡日期（共{len(rebalance_dates)}个）：")
for d in rebalance_dates:
    print(d.date(), end='  ')
    # 每10个换行，便于查看
    if list(rebalance_dates).index(d) % 10 == 9:
        print()
print()