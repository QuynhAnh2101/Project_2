import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import re
import plotly.graph_objects as go

# --- 1. CSS: CHỈNH LẠI KÍCH THƯỚC VỪA MẮT ---
st.set_page_config(page_title="AI Electricity", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    /* Khoảng cách các khối vừa đủ, không quá khít cũng không quá thưa */
    [data-testid="stVerticalBlock"] { gap: 1rem !important; } 
    
    /* KHUNG KẾT QUẢ: Tinh tế và nổi bật vừa đủ */
    .result-container {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-top: 10px;
    }
    
    .result-label {
        color: #00f2ff;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0px;
    }
    
    .result-price {
        color: #ffffff !important;
        font-size: 3rem; /* Đã hạ xuống mức vừa phải, vẫn rất nổi */
        font-weight: 800;
        margin: 5px 0;
    }
    
    .result-unit {
        color: #00f2ff;
        font-size: 1.2rem;
    }

    .stButton>button {
        width: 100%;
        background: #00f2ff;
        color: #0d1117 !important;
        font-weight: bold;
        border: none;
        height: 45px;
        border-radius: 8px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MACHINE LEARNING (GIỮ NGUYÊN) ---
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv('data.csv')
        mapping = {'people': 'người', 'ac_h': 'máy lạnh', 'fans': 'quạt', 
                   'fan_h': 'giờ quạt', 'fridge': 'tủ lạnh', 'cook': 'bếp', 
                   'pc': 'máy tính', 'price': 'giá', 'target': 'tiền'}
        processed = pd.DataFrame()
        for key, pattern in mapping.items():
            col = next((c for c in df.columns if pattern.lower() in c.lower()), None)
            if col:
                if key in ['fridge', 'cook', 'pc']:
                    processed[key] = df[col].apply(lambda x: 1 if any(w in str(x).lower() for w in ['có', 'dùng', 'yes']) else 0)
                else:
                    processed[key] = df[col].apply(lambda x: float(re.findall(r'\d+', str(x).replace('.', ''))[0]) if re.findall(r'\d+', str(x)) else 0.0)
        X, y = processed.drop('target', axis=1), processed['target']
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
        return model, X.columns.tolist()
    except: return None, None

model, features = load_and_train()

# --- 3. UI RENDER ---
st.title("⚡HỆ THỐNG DỰ ĐOÁN TIỀN ĐIỆN⚡")

if model:
    col_in, col_out = st.columns([1, 1.4], gap="large")
    
    with col_in:
        st.write("### 🏠 Nhập thông tin")
        p_people = st.number_input("Số người ở", min_value=1, value=1)
        p_price = st.selectbox("Giá điện (đ/kWh)", [2500, 3000, 3500, 4000], index=None, placeholder="Chọn mức giá...")
        
        st.write("---")
        p_ac_h = st.slider("Giờ máy lạnh/ngày", 0, 24, 0)
        p_fan_n = st.number_input("Số lượng quạt", 0, 10, 0)
        p_fan_h = st.slider("Giờ quạt/ngày", 0, 24, 0)
        
        st.write("---")
        c1, c2, c3 = st.columns(3)
        p_fridge, p_cook, p_pc = c1.checkbox("Tủ lạnh"), c2.checkbox("Bếp điện"), c3.checkbox("Máy tính")
        
        predict_btn = st.button("🚀 DỰ ĐOÁN NGAY")

    with col_out:
        if predict_btn and p_price is not None:
            user_vals = {'people': p_people, 'ac_h': p_ac_h, 'fans': p_fan_n, 'fan_h': p_fan_h, 
                         'fridge': int(p_fridge), 'cook': int(p_cook), 'pc': int(p_pc), 'price': p_price}
            input_row = [user_vals.get(f, 0) for f in features]
            pred = model.predict([input_row])[0]
            
            # GIÁ TIỀN VỪA VẶN - SANG TRỌNG
            st.markdown(f"""
                <div class="result-container">
                    <p class="result-label">CHI PHÍ DỰ KIẾN</p>
                    <p class="result-price">{pred:,.0f} <span class="result-unit">VNĐ</span></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Biểu đồ
            kwh = {"Máy lạnh": p_ac_h*22, "Quạt": p_fan_n*p_fan_h*1.5, "Tủ lạnh": 35 if p_fridge else 0, "Bếp": 40 if p_cook else 0, "PC": 20 if p_pc else 0}
            plot_data = {k: v for k, v in kwh.items() if v > 0}
            fig = go.Figure(go.Bar(x=list(plot_data.values()), y=list(plot_data.keys()), orientation='h', marker_color='#00f2ff'))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9", height=300, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.balloons()
        else:
            st.info("Nhập đầy đủ dữ liệu rồinhấn nút dự đoán nhé!")
else:
    st.error("Thiếu file dữ liệu!")