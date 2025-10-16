import requests
import pandas as pd
import smtplib
import numpy as np
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import time
import os
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# إعدادات API
API_KEY = "qT9W9kW3aYOSEeEsK92PpX"
BASE_URL = "https://fcsapi.com/api-v3/forex/history"

# -----------------------------
# إعدادات البريد
EMAIL_FROM = os.getenv('EMAIL_FROM', 'gptmoh5@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
EMAIL_TO = os.getenv('EMAIL_TO', 'gptmoh5@gmail.com')

# -----------------------------
# إعدادات التداول
SYMBOL = "EUR/USD"
TIMEFRAMES = {
    "1m": {"candles": 100, "weight": 1.0, "prediction_count": 5},
    "5m": {"candles": 120, "weight": 1.2, "prediction_count": 3},
    "15m": {"candles": 100, "weight": 1.4, "prediction_count": 2},
    "30m": {"candles": 80, "weight": 1.5, "prediction_count": 2},
    "1h": {"candles": 60, "weight": 1.6, "prediction_count": 2},
    "4h": {"candles": 40, "weight": 1.8, "prediction_count": 1}
}

# -----------------------------
# دالة جلب البيانات التاريخية
def get_historical_data(symbol, timeframe, count):
    try:
        url = f"{BASE_URL}?symbol={symbol}&period={timeframe}&access_key={API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('status') == True:
            candles = data['response']
            df = pd.DataFrame(candles)
            df['t'] = pd.to_datetime(df['t'])
            df = df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            })
            df = df[['t', 'open', 'high', 'low', 'close', 'volume']]
            return df.tail(count)
        return None
    except Exception as e:
        print(f"خطأ في جلب البيانات: {e}")
        return None

# -----------------------------
# دالة حساب المؤشرات الفنية للتنبؤ
def calculate_prediction_features(df):
    try:
        # المتوسطات المتحركة
        df['sma_5'] = ta.trend.SMAIndicator(df['close'], window=5).sma_indicator()
        df['sma_10'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
        df['ema_8'] = ta.trend.EMAIndicator(df['close'], window=8).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        
        # RSI متعدد
        df['rsi_6'] = ta.momentum.RSIIndicator(df['close'], window=6).rsi()
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # الاتجاه والقوة
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['momentum'] = ta.momentum.ROCIndicator(df['close'], window=5).roc()
        
        # أنماط الشموع
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_ratio'] = df['body'] / (df['high'] - df['low'])
        
        # حجم التداول النسبي
        df['volume_sma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    except Exception as e:
        print(f"خطأ في حساب المؤشرات: {e}")
        return df

# -----------------------------
# دالة التنبؤ بالشموع القادمة
def predict_next_candles(df, timeframe, prediction_count):
    try:
        # إعداد البيانات للتدريب
        features_df = calculate_prediction_features(df)
        
        # إنشاء الهدف (الاتجاه للشمعة التالية)
        features_df['next_close'] = features_df['close'].shift(-1)
        features_df['target'] = np.where(
            features_df['next_close'] > features_df['close'], 1, 0  # 1: صاعد, 0:هابط
        )
        
        # إعداد الميزات
        feature_columns = [
            'sma_5', 'sma_10', 'ema_8', 'ema_21', 'rsi_6', 'rsi_14',
            'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d',
            'bb_width', 'adx', 'momentum', 'body_ratio', 'volume_ratio'
        ]
        
        # تنظيف البيانات
        features_df = features_df.dropna()
        
        if len(features_df) < 30:
            return []
        
        X = features_df[feature_columns].values
        y = features_df['target'].values
        
        # تدريب النموذج
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # التنبؤ بالشموع القادمة
        predictions = []
        current_features = X[-1:].copy()
        
        for i in range(prediction_count):
            pred = model.predict(current_features)[0]
            proba = model.predict_proba(current_features)[0]
            confidence = max(proba) * 100
            
            predictions.append({
                'candle_number': i + 1,
                'direction': '🟢 صاعد' if pred == 1 else '🔴 هابط',
                'confidence': confidence,
                'timeframe': timeframe
            })
            
            # تحديث الميزات للتنبؤ التالي (محاكاة)
            if i < prediction_count - 1:
                current_features[0][0] *= 1.001 if pred == 1 else 0.999  # تحديث SMA
                current_features[0][4] = min(100, current_features[0][4] * 1.01) if pred == 1 else max(0, current_features[0][4] * 0.99)  # RSI
        
        return predictions
        
    except Exception as e:
        print(f"خطأ في التنبؤ: {e}")
        return []

# -----------------------------
# دالة تحليل الاتجاه الحالي
def analyze_current_trend(df, timeframe):
    try:
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # اتجاه قصير المدى (آخر 5 شموع)
        short_trend = "صاعد" if current['close'] > df['close'].iloc[-5] else "هابط"
        
        # اتجاه المتوسطات
        ma_trend = "صاعد" if current['sma_5'] > current['sma_10'] else "هابط"
        
        # قوة الاتجاه
        trend_strength = "قوي" if current['adx'] > 25 else "متوسط" if current['adx'] > 20 else "ضعيف"
        
        # حالة RSI
        rsi_status = "شراء مفرط" if current['rsi_14'] > 70 else "بيع مفرط" if current['rsi_14'] < 30 else "محايد"
        
        return {
            'timeframe': timeframe,
            'short_trend': short_trend,
            'ma_trend': ma_trend,
            'trend_strength': trend_strength,
            'rsi_status': rsi_status,
            'current_price': current['close'],
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        print(f"خطأ في تحليل الاتجاه: {e}")
        return None

# -----------------------------
# دالة إرسال تقرير الشموع المتوقع
def send_candle_prediction_report(symbol, trend_analysis, predictions):
    try:
        subject = f"📊 توقعات الشموع القادمة - {symbol} - {datetime.now().strftime('%H:%M')}"
        
        # بناء التقرير
        report = f"""
🎯 **تقرير توقعات الشموع القادمة**
────────────────────
📈 الزوج: {symbol}
🕐 وقت التحديث: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

📊 **الاتجاهات الحالية:**
"""
        
        # إضافة تحليل كل timeframe
        for analysis in trend_analysis:
            if analysis:
                report += f"""
⏰ **{analysis['timeframe']}:**
   • الاتجاه القصير: {analysis['short_trend']}
   • اتجاه المتوسطات: {analysis['ma_trend']}
   • قوة الاتجاه: {analysis['trend_strength']}
   • حالة RSI: {analysis['rsi_status']}
   • السعر: {analysis['current_price']:.5f}
"""
        
        report += f"\n🔮 **توقعات الشموع القادمة:**\n"
        
        # إضافة تنبؤات كل timeframe
        for pred in predictions:
            if pred:
                report += f"\n⏰ **{pred['timeframe']}:**\n"
                for candle_pred in pred['predictions']:
                    report += f"   • الشمعة {candle_pred['candle_number']}: {candle_pred['direction']} (ثقة: {candle_pred['confidence']:.1f}%)\n"
        
        report += """
────────────────────
💡 **توجيهات استراتيجية:**
• استخدم التوقعات كدعم لاتخاذ القرار
• الجمع بين multiple timeframes يزيد الدقة
• دائماً استخدم Stop Loss

📧 نظام التوقعات الذكي - بركة الله في تجارتك
"""
        
        msg = MIMEText(report, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        
        print(f"✅ تم إرسال تقرير التوقعات")
        return True
        
    except Exception as e:
        print(f"❌ خطأ في إرسال التقرير: {e}")
        return False

# -----------------------------
# الدالة الرئيسية للتوقعات
def run_candle_prediction_analysis():
    print(f"\n🔮 بدء تحليل توقعات الشموع لـ {SYMBOL} - {datetime.now()}")
    
    all_trend_analysis = []
    all_predictions = []
    
    for timeframe, config in TIMEFRAMES.items():
        try:
            print(f"   ⏳ جاري تحليل {timeframe}...")
            
            # جلب البيانات
            df = get_historical_data(SYMBOL, timeframe, config['candles'])
            if df is None or len(df) < 30:
                continue
            
            # حساب المؤشرات
            df = calculate_prediction_features(df)
            df = df.dropna()
            
            if len(df) < 20:
                continue
            
            # تحليل الاتجاه الحالي
            trend_analysis = analyze_current_trend(df, timeframe)
            if trend_analysis:
                all_trend_analysis.append(trend_analysis)
            
            # التنبؤ بالشموع القادمة
            predictions = predict_next_candles(df, timeframe, config['prediction_count'])
            if predictions:
                all_predictions.append({
                    'timeframe': timeframe,
                    'predictions': predictions
                })
            
            print(f"   ✅ تم تحليل {timeframe}: {len(predictions)} توقعات")
            
        except Exception as e:
            print(f"   ❌ خطأ في {timeframe}: {e}")
    
    # إرسال التقرير كل 15 دقيقة أو عندما تكون هناك تغييرات كبيرة
    if all_predictions and all_trend_analysis:
        send_candle_prediction_report(SYMBOL, all_trend_analysis, all_predictions)
        return True
    
    return False

# -----------------------------
# الحلقة الرئيسية
print("🚀 بدء نظام توقعات الشموع - بإذن الله توقعات دقيقة")
print("⏰ سيتم تحديث التوقعات كل دقيقة وإرسال التقارير كل 15 دقيقة")

last_report_time = datetime.now()
report_interval = 15  # دقائق

while True:
    try:
        current_time = datetime.now()
        
        # تشغيل التحليل كل دقيقة
        run_candle_prediction_analysis()
        
        # إرسال تقرير مفصل كل 15 دقيقة
        if (current_time - last_report_time).total_seconds() >= report_interval * 60:
            print(f"\n📨 إرسال التقرير الدوري كل {report_interval} دقيقة")
            run_candle_prediction_analysis()
            last_report_time = current_time
        
        print(f"⏳ انتظار دقيقة للتحديث التالي... ({current_time.strftime('%H:%M:%S')})")
        time.sleep(60)
        
    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف نظام التوقعات - بارك الله فيك")
        break
    except Exception as e:
        print(f"🔥 خطأ غير متوقع: {e}")
        time.sleep(60)
