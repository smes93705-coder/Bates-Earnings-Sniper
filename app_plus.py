import streamlit as st
import QuantLib as ql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import yfinance as yf
import yahooquery as yq
from scipy.optimize import differential_evolution
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import time
import requests

# ==========================================
# 0. 基礎設定與中文化
# ==========================================
st.set_page_config(page_title="Bates 財報狂徒", page_icon="⚡", layout="wide")

# 設定中文字型 (避免圖表亂碼)
# 根據作業系統自動選擇最佳字體，確保雲端與本地都能顯示中文
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
# 1. 雙核心資料抓取模組 (Dual-Engine Fetcher)
# ==========================================

def get_session():
    """
    建立偽裝成瀏覽器的 Session，用來繞過 Yahoo Finance 的機器人阻擋
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    })
    return s


@st.cache_data(ttl=3600)
def get_valid_dates(ticker):
    """
    取得該股票所有選擇權到期日
    優先使用 yfinance，若失敗則自動切換至 yahooquery
    """
    # 嘗試引擎 A: yfinance
    try:
        stock = yf.Ticker(ticker, session=get_session())
        dates = stock.options
        if dates: return list(dates)
    except:
        pass

    # 嘗試引擎 B: yahooquery (備援)
    try:
        t = yq.Ticker(ticker)
        dates = t.options
        # yahooquery 有時回傳格式不同(dict/list)，需轉換
        if dates and isinstance(dates, dict):
            return list(dates.keys())
        if dates and not isinstance(dates, pd.DataFrame):
            return [str(d) for d in dates]
    except:
        pass

    return []


@st.cache_data(ttl=300)
def get_market_data(ticker, expiry_date, use_demo=False):
    """
    抓取 Spot Price, Option Chain, 並計算 IV Rank, MA 技術指標
    含：Demo 模式、雙引擎切換、資料清洗、EM 計算
    """
    fetch_time = datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")

    # --- 🧪 Demo 模式 (當 API 全掛時的緊急備援方案) ---
    if use_demo:
        spot = 100.0
        # 模擬一個標準的財報前微笑曲線 (Smile)
        strikes = np.linspace(80, 120, 40)
        # 模擬波動率：價外高，價平低
        vols = 0.5 + 0.015 * (strikes - 100) ** 2
        prices = []
        for k, v in zip(strikes, vols):
            # 簡單生成假價格 (Put)
            intrinsic = max(0, 100 - k)
            time_val = (100 * v * 0.1)  # 粗略估計
            prices.append(intrinsic + time_val * np.exp(-0.1 * abs(k - 100)))

        df = pd.DataFrame({
            'Strike': strikes, 'ImpliedVol': vols, 'MarketPrice': prices, 'Type': 'Put'
        })
        # 複製一份給 Call
        df2 = df.copy();
        df2['Type'] = 'Call'
        df = pd.concat([df, df2])

        extra = {
            "HV": 0.4,
            "ExpectedMove": 8.5,
            "ExpectedMovePct": 0.085,
            "ATM_IV": 0.5,
            "MA20": 105.0,
            "MA240": 90.0,  # 模擬多頭排列
            "Source": "🧪 虛擬演示數據 (API 失效時使用)"
        }
        return spot, df, fetch_time, extra

    # --- 真實數據抓取 (Real Data) ---
    spot = None
    puts_df = pd.DataFrame()
    calls_df = pd.DataFrame()
    source_name = "Unknown"

    ma20, ma240, hv_current = None, None, 0.4  # 預設值

    # 1. 嘗試 yfinance (主要引擎)
    try:
        stock = yf.Ticker(ticker, session=get_session())
        # 抓取 2 年歷史以計算年線 (MA240)
        hist = stock.history(period="2y")
        if not hist.empty:
            spot = float(hist['Close'].iloc[-1])
            opt = stock.option_chain(expiry_date)
            puts = opt.puts
            calls = opt.calls

            # 清洗與計算 Mid Price
            for d in [puts, calls]:
                # 補零防呆
                d.fillna(0, inplace=True)
                # 計算中價: (Bid+Ask)/2，若無則用 Last
                d['Mid'] = np.where((d['bid'] > 0) & (d['ask'] > 0), (d['bid'] + d['ask']) / 2, d['lastPrice'])

            puts_df = pd.DataFrame(
                {'Strike': puts['strike'], 'IV': puts['impliedVolatility'], 'Price': puts['Mid'], 'Type': 'Put'})
            calls_df = pd.DataFrame(
                {'Strike': calls['strike'], 'IV': calls['impliedVolatility'], 'Price': calls['Mid'], 'Type': 'Call'})
            source_name = "Yahoo Finance (Primary)"

            # 計算技術指標
            if len(hist) >= 20: ma20 = hist['Close'].rolling(20).mean().iloc[-1]
            if len(hist) >= 240: ma240 = hist['Close'].rolling(240).mean().iloc[-1]
            hv_current = np.log(hist['Close'] / hist['Close'].shift(1)).std() * np.sqrt(252)
    except:
        pass

    # 2. 若失敗，嘗試 yahooquery (備援引擎)
    if spot is None or puts_df.empty:
        try:
            t = yq.Ticker(ticker)
            price_data = t.price
            spot = price_data[ticker]['regularMarketPrice']

            # 嘗試抓歷史 (yahooquery history)
            hist = t.history(period='2y')
            if not hist.empty:
                if isinstance(hist.index, pd.MultiIndex):
                    hist = hist.reset_index().set_index('date')
                if 'close' in hist.columns:
                    ma20 = hist['close'].rolling(20).mean().iloc[-1]
                    ma240 = hist['close'].rolling(240).mean().iloc[-1]
                    hv_current = np.log(hist['close'] / hist['close'].shift(1)).std() * np.sqrt(252)

            # 抓選擇權
            opts = t.option_chain
            if isinstance(opts, pd.DataFrame):
                opts = opts.reset_index()
                target_str = expiry_date.strftime('%Y-%m-%d')
                opts['expiration'] = opts['expiration'].astype(str)
                day_opts = opts[opts['expiration'] == target_str]

                if not day_opts.empty:
                    p_raw = day_opts[day_opts['optionType'] == 'puts']
                    c_raw = day_opts[day_opts['optionType'] == 'calls']

                    for d in [p_raw, c_raw]:
                        d['Mid'] = (d['bid'] + d['ask']) / 2
                        d['Mid'] = d['Mid'].fillna(d['lastPrice'])

                    puts_df = pd.DataFrame(
                        {'Strike': p_raw['strike'], 'IV': p_raw['impliedVolatility'], 'Price': p_raw['Mid'],
                         'Type': 'Put'})
                    calls_df = pd.DataFrame(
                        {'Strike': c_raw['strike'], 'IV': c_raw['impliedVolatility'], 'Price': c_raw['Mid'],
                         'Type': 'Call'})
                    source_name = "YahooQuery (Backup)"
        except:
            pass

    # 若兩者皆失敗
    if spot is None or puts_df.empty:
        return None, None, fetch_time, None

    # --- 計算關鍵指標：Expected Move (EM) ---
    # 公式： (ATM Call + ATM Put) * 0.85
    atm_strike = min(puts_df['Strike'], key=lambda x: abs(x - spot))
    try:
        atm_call = calls_df[calls_df['Strike'] == atm_strike]['Price'].values[0]
        atm_put = puts_df[puts_df['Strike'] == atm_strike]['Price'].values[0]
        em = (atm_call + atm_put) * 0.85
    except:
        em = spot * 0.05  # Fallback: 若無數據則假設 5%

    # 數據合併與篩選 (保留 50% ~ 150% 範圍，確保紅線能畫出來)
    df = pd.concat([puts_df[puts_df['Strike'] < spot], calls_df[calls_df['Strike'] > spot]])
    df = df[(df['Strike'] > spot * 0.5) & (df['Strike'] < spot * 1.5)]
    # 只保留有成交或有報價的數據，避免雜訊干擾模型
    df = df[(df['Price'] > 0.01)].sort_values('Strike')

    extra = {
        "HV": hv_current,
        "ExpectedMove": em,
        "ExpectedMovePct": em / spot,
        "ATM_IV": puts_df[puts_df['Strike'] == atm_strike]['IV'].mean(),
        "MA20": ma20,
        "MA240": ma240,
        "Source": source_name
    }

    return spot, df, fetch_time, extra


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

        # 參數初始化 (Bates 模型的 8 個參數)
        # v0: 當前變異數, theta: 長期均值, kappa: 回歸速度, sigma: Vol of Vol, rho: 相關係數
        # lambda: 跳躍頻率, nu: 跳躍均值, delta: 跳躍標準差
        self.v0 = 0.04;
        self.theta = 0.04;
        self.kappa = 1.0;
        self.sigma = 0.5;
        self.rho = -0.5
        self.lambda_jump = 0.1;
        self.nu_jump = -0.1;
        self.delta_jump = 0.1
        self.helpers = []

    def setup_helpers(self, market_data, expiry_date):
        self.helpers = []
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        ql_expiry = ql.Date(expiry_date.day, expiry_date.month, expiry_date.year)

        # 避免 T=0
        days = (ql_expiry - self.calculation_date)
        period = ql.Period(max(1, days), ql.Days)

        # 鎖定 V0 (利用 ATM IV 的平方作為起點，加速收斂)
        try:
            spot_val = self.spot.value()
            closest_idx = (market_data['Strike'] - spot_val).abs().idxmin()
            val = market_data.loc[closest_idx, 'IV']
            if val > 0: self.v0 = float(val) ** 2; self.theta = self.v0
        except:
            pass

        for _, row in market_data.iterrows():
            helper = ql.HestonModelHelper(
                period, calendar, self.spot.value(), float(row['Strike']),
                ql.QuoteHandle(ql.SimpleQuote(float(row['Price']))),
                self.risk_free_ts, self.dividend_ts,
                ql.HestonModelHelper.PriceError
            )
            self.helpers.append(helper)

    def cost_function(self, params):
        # 最小化誤差函數 (RMSE)
        k, s, r, l, n, d = params
        try:
            process = ql.BatesProcess(self.risk_free_ts, self.dividend_ts, self.spot, self.v0, k, self.theta, s, r, l,
                                      n, d)
            engine = ql.BatesEngine(ql.BatesModel(process))
            error = 0.0
            for h in self.helpers:
                h.setPricingEngine(engine)
                mkt = h.marketValue()
                # 加權誤差：價外權重較低
                weight = 1.0 / (mkt + 0.5)
                error += ((h.modelValue() - mkt) * weight) ** 2
            return error
        except:
            return 1e9

    def calibrate(self):
        # 差分進化演算法 (Differential Evolution) 尋找最佳解
        bounds = [
            (0.1, 5.0), (0.01, 2.0), (-0.95, 0.95),  # Heston Params
            (0.01, 5.0), (-0.3, 0.3), (0.01, 0.3)  # Jump Params
        ]
        try:
            res = differential_evolution(self.cost_function, bounds, strategy='best1bin', maxiter=5, popsize=6, seed=42)
            self.kappa, self.sigma, self.rho, self.lambda_jump, self.nu_jump, self.delta_jump = res.x
        except:
            pass  # 若失敗則沿用初始值

        return {
            "v0": self.v0, "kappa": self.kappa, "theta": self.theta, "sigma": self.sigma,
            "rho": self.rho, "lambda": self.lambda_jump, "nu": self.nu_jump, "delta": self.delta_jump
        }


# ==========================================
# 3. 風險分析 (Risk Engine)
# ==========================================
def analyze_risk(spot, rf, div, expiry, params, otype, extra):
    ql_expiry = ql.Date(expiry.day, expiry.month, expiry.year)
    today = ql.Date.todaysDate()
    T = max(1e-4, (ql_expiry - today) / 365.0)

    spot_h = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, float(rf), ql.Actual365Fixed()))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, float(div), ql.Actual365Fixed()))

    # Bates 引擎
    proc = ql.BatesProcess(r_ts, q_ts, spot_h, params['v0'], params['kappa'], params['theta'], params['sigma'],
                           params['rho'], params['lambda'], params['nu'], params['delta'])
    eng = ql.BatesEngine(ql.BatesModel(proc))

    # BS 引擎 (對照組)
    bs_vol = np.sqrt(params['v0'])
    bs_proc = ql.BlackScholesMertonProcess(spot_h, q_ts, r_ts, ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.UnitedStates(ql.UnitedStates.NYSE), bs_vol, ql.Actual365Fixed())))
    bs_eng = ql.AnalyticEuropeanEngine(bs_proc)

    results = []
    # 掃描範圍設定
    if otype == "put":
        scan = np.arange(int(spot * 0.6), int(spot), max(1, int(spot * 0.01)))
    else:
        scan = np.arange(int(spot), int(spot * 1.4), max(1, int(spot * 0.01)))
    if len(scan) == 0: scan = [spot]

    delta_k = 0.05
    try:
        disc = r_ts.discount(ql_expiry)
    except:
        disc = 1.0

    for strike in sorted(scan, reverse=(otype == "put")):
        p_p = ql.PlainVanillaPayoff(ql.Option.Put if otype == "put" else ql.Option.Call, float(strike) + delta_k)
        p_m = ql.PlainVanillaPayoff(ql.Option.Put if otype == "put" else ql.Option.Call, float(strike) - delta_k)
        ex = ql.EuropeanExercise(ql_expiry)

        # 1. Bates 真實機率
        op_h_p = ql.VanillaOption(p_p, ex);
        op_h_p.setPricingEngine(eng)
        op_h_m = ql.VanillaOption(p_m, ex);
        op_h_m.setPricingEngine(eng)
        h_prob = abs((op_h_p.NPV() - op_h_m.NPV()) / (2 * delta_k * disc))

        # 2. BS 機率 (虛假機率)
        op_b_p = ql.VanillaOption(p_p, ex);
        op_b_p.setPricingEngine(bs_eng)
        op_b_m = ql.VanillaOption(p_m, ex);
        op_b_m.setPricingEngine(bs_eng)
        b_prob = abs((op_b_p.NPV() - op_b_m.NPV()) / (2 * delta_k * disc))

        # 3. Delta
        d1 = (np.log(spot / strike) + (rf - div + 0.5 * bs_vol ** 2) * T) / (bs_vol * np.sqrt(T))
        delta = norm.cdf(d1) if otype == "call" else norm.cdf(d1) - 1

        # 4. 安全分數 (EM)
        safe = abs((strike - spot) / spot) / extra['ExpectedMovePct'] if extra['ExpectedMovePct'] > 0 else 0

        # 5. 評估邏輯
        status, lvl = "⭕ 普通", 1
        if h_prob > 0.15:
            status, lvl = "💀 危險 (Avoid)", 3
        elif safe < 1.0:
            status, lvl = "❌ 射程內 (Risky)", 2
        elif h_prob < 0.08 and safe > 1.2:
            status, lvl = "✅ 甜蜜點 (Sweet)", 0
        elif (h_prob - b_prob) > 0.05:
            status, lvl = "⚠️ 肥尾", 2

        results.append(
            {"Strike": strike, "Dist%": (strike - spot) / spot, "Dist(EM)": safe, "Delta": delta, "BS_Prob": b_prob,
             "Bates_Prob": h_prob, "Eval": status, "Lvl": lvl})

    return pd.DataFrame(results)


# ==========================================
# 4. 前端介面 (UI)
# ==========================================
with st.sidebar:
    st.header("⚙️ 1. 參數設定")
    ticker = st.text_input("股票代碼", "NVDA").upper()

    dates = get_valid_dates(ticker)
    expiry_date = None
    if dates:
        idx = 1 if len(dates) > 1 else 0
        expiry_str = st.selectbox("到期日", dates, index=idx)
        expiry_date = pd.to_datetime(expiry_str)
    else:
        st.warning("⚠️ 連線受阻，請嘗試演示模式")

    st.markdown("---")
    st.header("⚙️ 2. 環境參數")
    rf = st.number_input("無風險利率", 4.5) / 100
    div = st.number_input("股利率", 0.0) / 100

    c1, c2 = st.columns(2)
    with c1:
        run_btn = st.button("⚡ 執行", type="primary")
    with c2:
        demo_btn = st.button("🧪 演示")

st.title("⚡ Bates 財報狂徒")

tab1, tab2 = st.tabs(["🚀 戰情室", "📚 戰略手冊"])

with tab1:
    if (run_btn and expiry_date) or demo_btn:
        is_demo = True if demo_btn else False
        msg = "正在使用雙核心引擎連線..." if not is_demo else "正在生成演示數據..."

        with st.spinner(msg):
            spot, df_mk, time, extra = get_market_data(ticker, expiry_date, use_demo=is_demo)

            if spot:
                st.caption(f"數據時間: {time} | 來源: {extra.get('Source')} | 現價: ${spot:.2f}")
                cal = BatesCalibrator(ql.Date.todaysDate(), spot, rf, div)
                cal.setup_helpers(df_mk, expiry_date)
                params = cal.calibrate()

                # --- 1. 趨勢與技術防線 ---
                st.subheader("🚦 趨勢與技術防線")
                trend = "⚖️ 震盪"
                ma240 = extra.get('MA240')
                if ma240:
                    if spot > ma240:
                        trend = "📈 多頭 (股價 > 年線)"
                    else:
                        trend = "📉 空頭 (股價 < 年線)"
                    dist_ma = (spot - ma240) / ma240
                    ma_str = f"${ma240:.2f} (乖離: {dist_ma:.1%})"
                else:
                    ma_str = "無資料"

                c_t1, c_t2, c_t3 = st.columns(3)
                c_t1.metric("趨勢判讀", trend)
                c_t2.metric("年線 (MA240)", ma_str)
                # Lambda 警示
                lam = params['lambda']
                lam_msg = "✅ 正常" if lam < 1.0 else ("⚠️ 頻繁" if lam < 3.0 else "💀 極度危險")
                c_t3.metric("跳空強度 (Lambda)", f"{lam:.2f} ({lam_msg})")

                # --- 2. 風險指標 ---
                st.subheader("📊 選擇權風險指標")
                c1, c2, c3 = st.columns(3)
                c1.metric("現價", f"${spot:.2f}")
                c2.metric("EM (預期震幅)", f"±${extra['ExpectedMove']:.2f} ({extra['ExpectedMovePct']:.1%})",
                          help="莊家防守線")
                c3.metric("ATM IV", f"{extra['ATM_IV']:.1%}")

                # --- 3. 微笑曲線圖表 ---
                st.subheader("1. 波動率微笑 (Bates Fit)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_mk['Strike'], df_mk['IV'], 'bo', alpha=0.5, label='Market IV (市場數據)')
                ax.axvline(spot + extra['ExpectedMove'], color='gray', linestyle='--', label='EM 邊界 (危險區)')
                ax.axvline(spot - extra['ExpectedMove'], color='gray', linestyle='--')

                # 紅線修復 (Interpolation)
                m_k = np.linspace(df_mk['Strike'].min(), df_mk['Strike'].max(), 50)
                m_v = []
                proc = ql.BatesProcess(cal.risk_free_ts, cal.dividend_ts, cal.spot, params['v0'], params['kappa'],
                                       params['theta'], params['sigma'], params['rho'], params['lambda'], params['nu'],
                                       params['delta'])
                eng = ql.BatesEngine(ql.BatesModel(proc))
                per = ql.Period(
                    max(1, (ql.Date(expiry_date.day, expiry_date.month, expiry_date.year) - ql.Date.todaysDate())),
                    ql.Days)

                for k in m_k:
                    try:
                        h = ql.HestonModelHelper(per, ql.UnitedStates(ql.UnitedStates.NYSE), spot, k,
                                                 ql.QuoteHandle(ql.SimpleQuote(0.0)), cal.risk_free_ts, cal.dividend_ts,
                                                 ql.HestonModelHelper.ImpliedVolError)
                        h.setPricingEngine(eng)
                        px = h.modelValue()
                        if px > 0.001:
                            m_v.append(h.impliedVolatility(px, 1e-3, 1000, 0.001, 5.0))
                        else:
                            m_v.append(np.nan)
                    except:
                        m_v.append(np.nan)

                ax.plot(m_k, pd.Series(m_v).interpolate(limit_direction='both'), 'r-', label='Bates Model (理論)')
                ax.set_xlabel("履約價 (Strike)")
                ax.set_ylabel("隱含波動率 (IV)")
                ax.legend();
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.caption("⚠️ 若紅線不完整，代表深價外理論價格過低，不影響表格準確性。")

                # --- 4. 報表 ---
                st.subheader("2. 策略掃描")


                def c_risk(r):
                    if r['Lvl'] == 0: return ['background-color: #d4edda'] * len(r)
                    if r['Lvl'] == 3: return ['background-color: #f8d7da'] * len(r)
                    return [''] * len(r)


                t1, t2 = st.tabs(["Short Put", "Short Call"])
                with t1:
                    df = analyze_risk(spot, rf, div, expiry_date, params, "put", extra)
                    st.dataframe(df.style.apply(c_risk, axis=1).format(
                        {"Dist%": "{:.1%}", "Dist(EM)": "{:.1f}x", "Delta": "{:.2f}", "BS_Prob": "{:.1%}",
                         "Bates_Prob": "{:.1%}"}), use_container_width=True)
                with t2:
                    df = analyze_risk(spot, rf, div, expiry_date, params, "call", extra)
                    st.dataframe(df.style.apply(c_risk, axis=1).format(
                        {"Dist%": "{:.1%}", "Dist(EM)": "{:.1f}x", "Delta": "{:.2f}", "BS_Prob": "{:.1%}",
                         "Bates_Prob": "{:.1%}"}), use_container_width=True)
            else:
                st.error("無法取得數據，請使用演示模式。")

with tab2:
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
    ### 🛡️ 三大保命濾網 (下單前必看)
    1.  **🛡️ EM 護城河**：履約價必須 > 1.2 倍 Expected Move。這是莊家的防守線，千萬別站進去。
    2.  **📉 歷史慣性**：若歷史平均跳空 15%，而這次 EM 只有 5%，代表市場嚴重低估風險，**千萬別賣**。
    3.  **🧱 技術防線**：最好選擇在 **年線 (MA240)** 或 **整數關卡** 之外的履約價，多一層支撐。

    ---

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