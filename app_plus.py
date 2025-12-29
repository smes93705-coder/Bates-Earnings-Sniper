import streamlit as st
import QuantLib as ql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import yfinance as yf
from scipy.optimize import differential_evolution
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import time

# ==========================================
# 0. 基礎設定與中文化
# ==========================================
st.set_page_config(page_title="Bates 財報狂徒", page_icon="⚡", layout="wide")

# 設定中文字型 (避免圖表亂碼)
system_name = platform.system()
if system_name == "Windows":
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
elif system_name == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

tw_tz = pytz.timezone('Asia/Taipei')


# ==========================================
# 1. 強力資料抓取模組 (Robust Data Fetcher)
# ==========================================

@st.cache_data(ttl=3600)
def get_valid_dates(ticker):
    """
    取得該股票所有選擇權到期日 (含重試機制)
    """
    for _ in range(3):
        try:
            stock = yf.Ticker(ticker)
            dates = stock.options
            if dates: return list(dates)
            time.sleep(1)
        except:
            time.sleep(1)
    return []


@st.cache_data(ttl=300)
def get_market_data(ticker, expiry_date):
    """
    抓取 Spot Price, Option Chain, 並計算 IV Rank, MA 技術指標
    回傳: (現價, 整理後的選擇權表, 抓取時間, 額外資訊)
    """
    fetch_time = datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
    stock = yf.Ticker(ticker)

    # 1. 抓取現貨與歷史數據 (抓 2 年以計算年線)
    try:
        hist = stock.history(period="2y")
        if hist.empty: return None, None, fetch_time, None
        spot = float(hist['Close'].iloc[-1])

        # 計算歷史波動率 (HV)
        hist['LogRet'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hv_current = hist['LogRet'].std() * np.sqrt(252)

        # 計算技術指標 (MA20, MA240)
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1] if len(hist) >= 20 else None
        ma240 = hist['Close'].rolling(window=240).mean().iloc[-1] if len(hist) >= 240 else None

    except:
        return None, None, fetch_time, None

    # 2. 抓取選擇權鏈
    try:
        opt = stock.option_chain(expiry_date)
        puts = opt.puts
        calls = opt.calls
    except:
        return None, None, fetch_time, None

    # 3. 資料清洗 (處理 NaN)
    for df in [puts, calls]:
        for col in ['bid', 'ask', 'lastPrice', 'impliedVolatility', 'strike']:
            if col not in df.columns: df[col] = 0.0
        df.fillna(0, inplace=True)
        # 計算中價
        df['MidPrice'] = np.where(
            (df['bid'] > 0) & (df['ask'] > 0),
            (df['bid'] + df['ask']) / 2,
            df['lastPrice']
        )
        df['MarketPrice'] = df['MidPrice']

    # 整理 DataFrame
    puts_data = pd.DataFrame({
        'Strike': puts['strike'], 'ImpliedVol': puts['impliedVolatility'],
        'MarketPrice': puts['MarketPrice'], 'Type': 'Put'
    })
    calls_data = pd.DataFrame({
        'Strike': calls['strike'], 'ImpliedVol': calls['impliedVolatility'],
        'MarketPrice': calls['MarketPrice'], 'Type': 'Call'
    })

    # 4. 計算 ATM Straddle Price (市場預期震幅 Expected Move)
    atm_strike = min(puts_data['Strike'], key=lambda x: abs(x - spot))
    try:
        atm_call = calls_data[calls_data['Strike'] == atm_strike]['MarketPrice'].values[0]
        atm_put = puts_data[puts_data['Strike'] == atm_strike]['MarketPrice'].values[0]
        # 公式備註：ATM Straddle * 0.85
        expected_move_dollar = (atm_call + atm_put) * 0.85
    except:
        expected_move_dollar = spot * 0.05

    expected_move_pct = expected_move_dollar / spot

    # 5. 篩選 OTM (價外) 用於校準
    otm_puts = puts_data[puts_data['Strike'] < spot]
    otm_calls = calls_data[calls_data['Strike'] > spot]
    df = pd.concat([otm_puts, otm_calls]).reset_index(drop=True)

    # 過濾極端值以利畫圖
    df = df[(df['Strike'] > spot * 0.50) & (df['Strike'] < spot * 1.50)]
    df = df[(df['MarketPrice'] > 0.01) & (df['ImpliedVol'] > 0)].sort_values(by='Strike').reset_index(drop=True)

    extra_info = {
        "HV": hv_current,
        "ExpectedMove": expected_move_dollar,
        "ExpectedMovePct": expected_move_pct,
        "ATM_IV": (puts_data[puts_data['Strike'] == atm_strike]['ImpliedVol'].mean()),
        "MA20": ma20,
        "MA240": ma240
    }

    return spot, df, fetch_time, extra_info


# ==========================================
# 2. Bates 模型校準器 (核心數學引擎)
# ==========================================
class BatesCalibrator:
    def __init__(self, calculation_date, spot, risk_free_rate, dividend_yield):
        self.calculation_date = calculation_date
        ql.Settings.instance().evaluationDate = calculation_date
        self.spot = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
        self.risk_free_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, float(risk_free_rate), ql.Actual365Fixed()))
        self.dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, float(dividend_yield), ql.Actual365Fixed()))

        # 參數初始化
        self.v0 = 0.04
        self.theta = 0.04
        self.kappa = 1.0
        self.sigma = 0.5
        self.rho = -0.5
        self.lambda_jump = 0.1
        self.nu_jump = -0.1
        self.delta_jump = 0.1

        self.helpers = []

    def setup_helpers(self, market_data, expiry_date):
        self.helpers = []
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        ql_expiry = ql.Date(expiry_date.day, expiry_date.month, expiry_date.year)
        days = (ql_expiry - self.calculation_date)
        period = ql.Period(max(1, days), ql.Days)

        try:
            spot_val = self.spot.value()
            closest_idx = (market_data['Strike'] - spot_val).abs().idxmin()
            val = market_data.loc[closest_idx, 'ImpliedVol']
            if val > 0:
                self.v0 = float(val) ** 2
                self.theta = self.v0
        except:
            pass

        for _, row in market_data.iterrows():
            helper = ql.HestonModelHelper(
                period, calendar, self.spot.value(), float(row['Strike']),
                ql.QuoteHandle(ql.SimpleQuote(float(row['MarketPrice']))),
                self.risk_free_ts, self.dividend_ts,
                ql.HestonModelHelper.PriceError
            )
            self.helpers.append(helper)

    def cost_function(self, params):
        k, s, r, l, n, d = params
        try:
            process = ql.BatesProcess(self.risk_free_ts, self.dividend_ts, self.spot, self.v0, k, self.theta, s, r, l,
                                      n, d)
            engine = ql.BatesEngine(ql.BatesModel(process))
            error = 0.0
            for h in self.helpers:
                h.setPricingEngine(engine)
                mkt = h.marketValue()
                mod = h.modelValue()
                weight = 1.0 / (mkt + 0.5)
                error += ((mod - mkt) * weight) ** 2
            return error
        except:
            return 1e9

    def calibrate(self):
        bounds = [
            (0.1, 5.0), (0.01, 2.0), (-0.95, 0.95),
            (0.01, 5.0), (-0.3, 0.3), (0.01, 0.3)
        ]
        try:
            res = differential_evolution(self.cost_function, bounds, strategy='best1bin', maxiter=5, popsize=6, seed=42)
            k, s, r, l, n, d = res.x
            self.kappa, self.sigma, self.rho, self.lambda_jump, self.nu_jump, self.delta_jump = k, s, r, l, n, d
        except:
            st.warning("模型校準未完全收斂，將使用預設參數進行估算。")

        return {
            "v0": self.v0, "kappa": self.kappa, "theta": self.theta, "sigma": self.sigma,
            "rho": self.rho, "lambda": self.lambda_jump, "nu": self.nu_jump, "delta": self.delta_jump
        }


# ==========================================
# 3. 風險分析與指標計算
# ==========================================
def analyze_risk(spot, risk_free, dividend, expiry_date, params, option_type, extra_info):
    ql_expiry = ql.Date(expiry_date.day, expiry_date.month, expiry_date.year)
    today = ql.Date.todaysDate()

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, float(risk_free), ql.Actual365Fixed()))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, float(dividend), ql.Actual365Fixed()))

    # 建立 Bates 模型引擎
    process = ql.BatesProcess(r_ts, q_ts, spot_handle, params['v0'], params['kappa'], params['theta'], params['sigma'],
                              params['rho'], params['lambda'], params['nu'], params['delta'])
    engine = ql.BatesEngine(ql.BatesModel(process))

    # 建立 BS 模型引擎
    bs_vol = np.sqrt(params['v0'])
    bs_process = ql.BlackScholesMertonProcess(spot_handle, q_ts, r_ts, ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.UnitedStates(ql.UnitedStates.NYSE), bs_vol, ql.Actual365Fixed())))
    bs_engine = ql.AnalyticEuropeanEngine(bs_process)

    results = []

    # 設定掃描範圍
    if option_type == "put":
        start = int(spot * 0.6)
        end = int(spot)
        step = max(1, int(spot * 0.01))
        scan_strikes = np.arange(start, end, step)
    else:
        start = int(spot)
        end = int(spot * 1.4)
        step = max(1, int(spot * 0.01))
        scan_strikes = np.arange(start, end, step)

    if len(scan_strikes) == 0: scan_strikes = [spot]

    delta_k = 0.05
    try:
        discount = r_ts.discount(ql_expiry)
    except:
        discount = 1.0
    T = max(1e-4, (ql_expiry - today) / 365.0)

    for strike in sorted(scan_strikes, reverse=(option_type == "put")):
        exercise = ql.EuropeanExercise(ql_expiry)
        t_flag = ql.Option.Put if option_type == "put" else ql.Option.Call

        payoff_p = ql.PlainVanillaPayoff(t_flag, float(strike) + delta_k)
        payoff_m = ql.PlainVanillaPayoff(t_flag, float(strike) - delta_k)

        # 1. 計算 Bates 機率
        op_h_p = ql.VanillaOption(payoff_p, exercise);
        op_h_p.setPricingEngine(engine)
        op_h_m = ql.VanillaOption(payoff_m, exercise);
        op_h_m.setPricingEngine(engine)
        h_prob = abs((op_h_p.NPV() - op_h_m.NPV()) / (2 * delta_k * discount))

        # 2. 計算 BS 機率
        op_b_p = ql.VanillaOption(payoff_p, exercise);
        op_b_p.setPricingEngine(bs_engine)
        op_b_m = ql.VanillaOption(payoff_m, exercise);
        op_b_m.setPricingEngine(bs_engine)
        b_prob = abs((op_b_p.NPV() - op_b_m.NPV()) / (2 * delta_k * discount))

        # 3. 計算 Delta
        d1 = (np.log(spot / strike) + (risk_free - dividend + 0.5 * bs_vol ** 2) * T) / (bs_vol * np.sqrt(T))
        delta_val = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0

        # 4. 安全距離指標
        dist_pct = (strike - spot) / spot
        if extra_info['ExpectedMovePct'] > 0:
            safety_score = abs(dist_pct) / extra_info['ExpectedMovePct']
        else:
            safety_score = 0

        # 5. 戰略評估邏輯
        status = "⭕ 普通"
        risk_level = 1

        if h_prob > 0.15:
            status = "💀 危險 (Avoid)"
            risk_level = 3
        elif safety_score < 1.0:
            status = "❌ 射程內 (Risky)"
            risk_level = 2
        elif h_prob < 0.08 and safety_score > 1.2:
            status = "✅ 甜蜜點 (Sweet Spot)"
            risk_level = 0
        elif (h_prob - b_prob) > 0.05:
            status = "⚠️ 肥尾陷阱"

        results.append({
            "履約價": strike,
            "距離(%)": dist_pct,
            "距離(EM)": safety_score,
            "Delta": delta_val,
            "BS機率": b_prob,
            "Bates機率": h_prob,
            "評估": status,
            "RiskLevel": risk_level
        })

    return pd.DataFrame(results)


# ==========================================
# 4. 前端介面 (UI)
# ==========================================
with st.sidebar:
    st.header("⚙️ 1. 參數設定")
    ticker = st.text_input("股票代碼 (e.g. NVDA, TSLA)", "NVDA").upper()

    expiry_date = None
    dates = get_valid_dates(ticker)

    if dates:
        default_idx = 1 if len(dates) > 1 else 0
        expiry_str = st.selectbox("到期日 (建議選財報該週)", dates, index=default_idx)
        expiry_date = pd.to_datetime(expiry_str)
    else:
        st.error("❌ 找不到代碼或該股無選擇權資料 (ETF可能無資料)")

    st.markdown("---")
    st.header("⚙️ 2. 環境參數")
    risk_free = st.number_input("無風險利率 (4.5%)", value=4.5, step=0.1) / 100
    div_yield = st.number_input("股利率 (0%)", value=0.0, step=0.1) / 100

    run_btn = st.button("⚡ 執行策略分析", type="primary")

    st.markdown("---")
    st.caption("資料來源：Yahoo Finance API (延遲15分鐘)")
    st.info("💡 **小撇步**：財報季請尋找 Bates 機率 < 5% 的履約價。平日可放寬至 15%。")

st.title("⚡ Bates 財報狂徒")

# 分頁設計
tab_main, tab_edu = st.tabs(["🚀 策略分析儀表板", "📚 戰略指導手冊 & 註解"])

# --- 頁面 1: 主分析 ---
with tab_main:
    if run_btn and ticker and expiry_date:
        with st.spinner(f"正在連線 Yahoo Finance 抓取 {ticker} 選擇權數據..."):
            spot, df_market, fetch_time, extra = get_market_data(ticker, expiry_str)

            if df_market is None or df_market.empty:
                st.error("❌ 數據下載失敗，可能是 API 連線逾時，請稍後重試。")
                st.stop()

            # 校準
            st.caption(f"數據時間: {fetch_time} | 現價: ${spot:.2f} | 資料來源: Yahoo Finance")
            calibrator = BatesCalibrator(ql.Date.todaysDate(), spot, risk_free, div_yield)
            calibrator.setup_helpers(df_market, expiry_date)
            params = calibrator.calibrate()

            # --- 新增：趨勢與技術指標 ---
            st.subheader("🚦 趨勢與技術防線")

            trend_str = "⚖️ 震盪整理"
            ma240 = extra.get('MA240', None)
            if ma240:
                if spot > ma240:
                    trend_str = "📈 長線多頭 (股價 > 年線)"
                    trend_advice = "建議：順勢操作，可安心做 Short Put。"
                else:
                    trend_str = "📉 長線空頭 (股價 < 年線)"
                    trend_advice = "警告：逆勢操作，做 Short Put 請務必拉大安全距離。"
            else:
                trend_str = "⚠️ 資料不足 (無年線)"
                trend_advice = "建議觀望，或以 EM 指標為主。"

            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("目前趨勢判讀", trend_str, help="基於價格與年線(MA240)的關係")
            with col_t2:
                if ma240:
                    dist_ma = (spot - ma240) / ma240
                    st.metric("年線 (MA240) 點位", f"${ma240:.2f}", f"乖離率: {dist_ma:.1%}",
                              help="這是長線最強的技術支撐/壓力")
                else:
                    st.metric("年線 (MA240)", "計算中或數據不足")
            with col_t3:
                st.info(trend_advice)

            # --- 關鍵指標 ---
            st.subheader("📊 選擇權風險指標")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("現價 (Spot)", f"${spot:.2f}")
            with col2:
                iv_ratio = extra['ATM_IV'] / max(0.01, extra['HV'])
                st.metric("IV / HV 比率", f"{iv_ratio:.2f}x", help=">1.5 代表權利金昂貴，適合做賣方")
            with col3:
                st.metric("市場預期震幅 (EM)", f"±${extra['ExpectedMove']:.2f}", f"±{extra['ExpectedMovePct']:.1%}",
                          help="計算公式: (ATM Call + ATM Put) * 0.85。代表莊家的防守線。")
            with col4:
                # Lambda 說明追加數值判斷
                lambda_val = params['lambda']
                if lambda_val > 1.0:
                    lambda_status = "⚠️ 頻繁"
                elif lambda_val > 3.0:
                    lambda_status = "💀 極度危險"
                else:
                    lambda_status = "✅ 正常"

                st.metric("跳空強度 (Lambda)", f"{lambda_val:.2f} ({lambda_status})",
                          help="一年發生幾次大跳空？\n正常值：< 0.5\n警戒值：> 1.0\n危險值：> 3.0 (極度不穩)")

            # --- 圖表區 ---
            st.subheader("1. 波動率微笑 (Bates Fit)")
            col_chart, col_desc = st.columns([3, 1])

            with col_chart:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_market['Strike'], df_market['ImpliedVol'], 'bo', label='Market IV (市場數據)', alpha=0.5)
                ax.axvline(spot + extra['ExpectedMove'], color='gray', linestyle='--', label='EM 邊界 (危險區)')
                ax.axvline(spot - extra['ExpectedMove'], color='gray', linestyle='--')

                # 紅線修復
                min_k = df_market['Strike'].min()
                max_k = df_market['Strike'].max()
                model_strikes = np.linspace(min_k, max_k, 50)
                model_vols = []

                process = ql.BatesProcess(calibrator.risk_free_ts, calibrator.dividend_ts, calibrator.spot,
                                          params['v0'], params['kappa'], params['theta'], params['sigma'],
                                          params['rho'], params['lambda'], params['nu'], params['delta'])
                engine = ql.BatesEngine(ql.BatesModel(process))
                days = (ql.Date(expiry_date.day, expiry_date.month, expiry_date.year) - ql.Date.todaysDate())
                period = ql.Period(max(1, days), ql.Days)

                for k in model_strikes:
                    try:
                        h = ql.HestonModelHelper(period, ql.UnitedStates(ql.UnitedStates.NYSE), spot, k,
                                                 ql.QuoteHandle(ql.SimpleQuote(0.0)), calibrator.risk_free_ts,
                                                 calibrator.dividend_ts, ql.HestonModelHelper.ImpliedVolError)
                        h.setPricingEngine(engine)
                        price = h.modelValue()
                        if price > 0.001:
                            iv = h.impliedVolatility(price, 1e-3, 2000, 0.001, 5.0)
                            model_vols.append(iv)
                        else:
                            model_vols.append(np.nan)
                    except:
                        model_vols.append(np.nan)

                s_vols = pd.Series(model_vols).interpolate(limit_direction='both')
                ax.plot(model_strikes, s_vols, 'r-', label='Bates Model (理論曲線)', linewidth=2)

                ax.set_title(f"{ticker} 波動率微笑曲線 (Volatility Smile)")
                ax.set_xlabel("Strike Price (履約價)")
                ax.set_ylabel("Implied Volatility (隱含波動率)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.caption("⚠️ 紅線若未顯示，代表理論價格過低(Deep OTM)，不影響下方表格準確度。")

            with col_desc:
                st.markdown("""
                **圖表指南**：
                * **X軸**：履約價 (Strike)。
                * **Y軸**：隱含波動率 (IV)。
                * **紅線**：Bates 模型曲線。
                * **虛線**：EM 安全邊界。

                **操作**：
                選擇 **虛線之外** 的履約價。
                """)

            # --- 報表區 ---
            st.subheader("2. 策略掃描報表")


            def style_risk(row):
                if row['RiskLevel'] == 0: return ['background-color: #d4edda; color: #155724'] * len(row)
                if row['RiskLevel'] == 2: return ['background-color: #fff3cd; color: #856404'] * len(row)
                if row['RiskLevel'] == 3: return ['background-color: #f8d7da; color: #721c24'] * len(row)
                return [''] * len(row)


            t1, t2 = st.tabs(["📉 Short Put (做多支撐)", "📈 Short Call (做空壓力)"])

            with t1:
                df_put = analyze_risk(spot, risk_free, div_yield, expiry_date, params, "put", extra)
                d_show = df_put.copy()
                d_show['距離(%)'] = d_show['距離(%)'].apply(lambda x: f"{x:.1%}")
                d_show['距離(EM)'] = d_show['距離(EM)'].apply(lambda x: f"{x:.1f}x")
                d_show['Delta'] = d_show['Delta'].apply(lambda x: f"{x:.2f}")
                d_show['BS機率'] = d_show['BS機率'].apply(lambda x: f"{x:.1%}")
                d_show['Bates機率'] = d_show['Bates機率'].apply(lambda x: f"**{x:.1%}%**")
                st.dataframe(d_show.style.apply(style_risk, axis=1), use_container_width=True)

            with t2:
                df_call = analyze_risk(spot, risk_free, div_yield, expiry_date, params, "call", extra)
                d_show = df_call.copy()
                d_show['距離(%)'] = d_show['距離(%)'].apply(lambda x: f"+{x:.1%}")
                d_show['距離(EM)'] = d_show['距離(EM)'].apply(lambda x: f"{x:.1f}x")
                d_show['Delta'] = d_show['Delta'].apply(lambda x: f"{x:.2f}")
                d_show['BS機率'] = d_show['BS機率'].apply(lambda x: f"{x:.1%}")
                d_show['Bates機率'] = d_show['Bates機率'].apply(lambda x: f"**{x:.1%}%**")
                st.dataframe(d_show.style.apply(style_risk, axis=1), use_container_width=True)

# --- 頁面 2: 教學手冊 ---
with tab_edu:
    st.header("📚 戰略指導手冊")

    st.markdown("### 🚦 趨勢判讀與操作心法")
    st.markdown("""
    **如何判斷目前趨勢？**
    * **多頭 (Bull)**：當 **股價 > 年線 (MA240)**。這代表過去一年的平均持倉者都是賺錢的，下方有強力支撐。
        * **策略**：大膽做 **Short Put**。可以稍微激進一點選 Delta 0.15~0.2 的位置。
    * **空頭 (Bear)**：當 **股價 < 年線 (MA240)**。代表上方有層層套牢賣壓。
        * **策略**：做 **Short Put** 時務必保守！安全距離請拉大 (Dist > 1.5 EM)。或者考慮改做 Bear Call Spread。
    """)

    st.markdown("---")

    with st.expander("⚡ 極速下單流程 (省時版 S.O.P.)", expanded=True):
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.markdown("#### Step 1: 啟動")
            st.write("輸入代碼，查看 **Bates 機率 < 5%** 的履約價是哪一個？(例: $90 Put)")
        with col_s2:
            st.markdown("#### Step 2: 護城河 (EM)")
            st.write("檢查該履約價是否距離現價超過 **紅線虛線 (Expected Move)**？(Dist(EM) > 1.2)")
        with col_s3:
            st.markdown("#### Step 3: 防線")
            st.write("檢查該履約價是否在 **年線 (MA240)** 之下？如果在年線下，安全性倍增。")
        with col_s4:
            st.markdown("#### Step 4: 執行")
            st.write("建議使用 **Credit Spread (價差單)** 鎖定最大虧損。參考券商軟體報價下單。")

    st.markdown("---")

    st.markdown("""
    ### ⚔️ 世紀對決：Bates 模型 vs. BS 模型

    | 特徵 | 🔴 BS 模型 (Black-Scholes) | ⚡ Bates 模型 (Heston + Jump) |
    | :--- | :--- | :--- |
    | **假設** | 股價是連續的，像散步一樣 (幾何布朗運動)。 | 股價會 **「瞬間移動 (Jump)」**，且波動率會隨機變化。 |
    | **財報預測** | **完全失效**。認為崩盤 10% 的機率是 0%。 | **精準捕捉**。知道市場在害怕跳空，能算出真實風險。 |
    | **波動率** | 假設是定值 (Flat)。忽略了價外選擇權比較貴的事實。 | 完美擬合 **「波動率微笑 (Smile)」**。 |
    | **結論** | **玩具**。只能在平靜市場參考。 | **武器**。財報季、黑天鵝事件的必備工具。 |

    > **為什麼要選 Bates？**
    > 因為在財報季，**BS 模型會騙你**。它會告訴你：「這個履約價很遠，絕對安全 (機率 0.1%)」。
    > 但 **Bates 模型會警告你**：「市場權利金這麼貴，代表大家都在賭跳空，真實機率其實高達 10%！」
    > **聽 Bates 的，才能活得久。**

    ---

    ### 🛡️ 三大保命濾網 (下單前必看)
    1.  **🛡️ EM 護城河**：履約價必須 > 1.2 倍 Expected Move。這是莊家的防守線，千萬別站進去。
    2.  **📉 歷史慣性**：若歷史平均跳空 15%，而這次 EM 只有 5%，代表市場嚴重低估風險，**千萬別賣**。
    3.  **🧱 技術防線**：最好選擇在 **年線 (MA240)** 或 **整數關卡** 之外的履約價，多一層支撐。

    ---

    ### ⚖️ 財報季 vs 平日：操作心法總表

    | 項目 | 🔥 財報季 (Earnings Season) | 🌊 平日 (Regular Trading) |
    | :--- | :--- | :--- |
    | **核心風險** | **跳空風險 (Jump Risk)**：一翻兩瞪眼，可能直接穿價。 | **波動風險 (Vega Risk)**：股價緩跌，可透過轉倉防守。 |
    | **目標機率** | **Bates 機率 < 5% ~ 8%** (極度保守) | **Bates 機率約 15% ~ 20%** (約 16 Delta) |
    | **安全距離** | 必須 > **1.2 倍 EM** | 可視技術線圖支撐調整，約 **1.0 倍 EM** 即可。 |
    | **獲利來源** | 賺取「恐慌溢價」。市場定價 15% 機率，實際發生僅 8%。 | 賺取「時間價值 (Theta)」。利用高勝率長期累積獲利。 |
    | **模型選擇** | **必用 Bates** (捕捉跳空) | Bates 或 Heston 皆可 (防止低估肥尾)。 |

    ---

    ### 🧮 深度參數解析：Bates 模型的 8 個秘密
    這些參數不是冷冰冰的數字，它們代表了市場當下的情緒。

    #### 1. 基礎波動 (Heston 部分)
    * **V0 (初始變異數)**：現在市場有多恐慌？數值越高，權利金越貴。
    * **Theta (長期均值)**：恐慌過後，波動率會回到哪裡？若 V0 > Theta，代表短期恐慌。
    * **Kappa (回歸速度)**：恐慌消退多快？越高代表 IV Crush 越快，利於賣方。
    * **Sigma (波動率的波動率)**：市場情緒多神經質？高代表微笑曲線很陡，肥尾效應強。
    * **Rho (相關係數)**：股價跌的時候，恐慌會增加嗎？通常為負 (-0.7)，代表避險情緒重。

    #### 2. 跳躍風險 (Jump 部分)
    * **Lambda (跳躍強度)**：
        * **定義**：一年平均發生幾次崩盤/大跳空？
        * **標準**：正常 < 0.5，警戒 > 1.0，**危險 > 3.0 (極度不穩)**。
    * **Nu (跳躍均值)**：
        * **定義**：如果發生跳空，平均是漲還是跌？
        * **判讀**：負值 (e.g. -0.15) 代表市場預期崩盤；正值代表預期暴漲。
    * **Delta (跳躍標準差)**：跳空幅度的不確定性。
    """)