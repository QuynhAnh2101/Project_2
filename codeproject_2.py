import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import re
import plotly.graph_objects as go

# --- 1. CSS GIAO DIỆN: TẬP TRUNG VÀO GIÁ TIỀN ---
st.set_page_config(page_title="AI Electricity Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stVerticalBlock"] { gap: 1rem !important; } 
    
    .result-container {
        text-align: center;
        padding: 30px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-top: 10px;
    }
    
    .result-label { color: #00f2ff; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px; }
    .result-price { color: #ffffff !important; font-size: 3.5rem; font-weight: 800; margin: 10px 0; }
    .result-unit { color: #00f2ff; font-size: 1.5rem; }

    .stButton>button {
        width: 100%; background: #00f2ff; color: #0d1117 !important;
        font-weight: bold; border: none; height: 50px; border-radius: 8px;
    }
    
    /* Chỉnh cho các ô input gọn gàng */
    div[data-testid="stMarkdownContainer"] > p { font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MACHINE LEARNING BACKEND ---
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv('data.csv')
        # Map các cột dựa trên từ khóa trong file data của bà
        mapping = {
            'people': 'người', 'area': 'diện tích', 'ac_h': 'máy lạnh', 
            'fans': 'quạt', 'fan_h': 'giờ quạt', 'fridge': 'tủ lạnh', 
            'cook': 'bếp', 'pc': 'máy tính', 'price': 'giá', 
            'target_money': 'tiền điện'
        }
        processed = pd.DataFrame()
        for key, pattern in mapping.items():
            col = next((c for c in df.columns if pattern.lower() in c.lower()), None)
            if col:
                if key in ['fridge', 'cook', 'pc']:
                    processed[key] = df[col].apply(lambda x: 1 if any(w in str(x).lower() for w in ['có', 'dùng', 'yes']) else 0)
                else:
                    processed[key] = df[col].apply(lambda x: float(re.findall(r'\d+', str(x).replace('.', ''))[0]) if re.findall(r'\d+', str(x)) else 0.0)
        
        X = processed.drop(['target_money'], axis=1)
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, processed['target_money'])
        
        return model, X.columns.tolist()
    except: return None, None

model, features = load_and_train()

# --- 3. UI RENDER ---
st.title("⚡ HỆ THỐNG DỰ BÁO TIỀN ĐIỆN")

if model:
    col_in, col_out = st.columns([1, 1.4], gap="large")
    
    with col_in:
        st.write("### 🏠 Thông tin phòng")
        # Diện tích và Số người cùng một hàng
        c_p1, c_p2 = st.columns(2)
        p_people = c_p1.number_input("Số người ở", min_value=1, value=1)
        p_area = c_p2.number_input("Diện tích phòng (m²)", min_value=5, value=20)
        
        p_price = st.selectbox("Giá điện (đ/kWh)", [2500, 3000, 3500, 4000], index=None, placeholder="Chọn mức giá...")
        
        st.divider()
        p_ac_h = st.slider("Giờ máy lạnh/ngày", 0, 24, 0)
        p_fan_n = st.number_input("Số lượng quạt", 0, 10, 0)
        p_fan_h = st.slider("Giờ quạt/ngày", 0, 24, 0)
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        p_fridge, p_cook, p_pc = c1.checkbox("Tủ lạnh"), c2.checkbox("Bếp điện"), c3.checkbox("Máy tính")
        
        predict_btn = st.button("🚀 DỰ BÁO NGAY")

    with col_out:
        if predict_btn and p_price is not None:
            # Thu thập dữ liệu input
            user_vals = {'people': p_people, 'area': p_area, 'ac_h': p_ac_h, 'fans': p_fan_n, 
                         'fan_h': p_fan_h, 'fridge': int(p_fridge), 'cook': int(p_cook), 
                         'pc': int(p_pc), 'price': p_price}
            input_row = [user_vals.get(f, 0) for f in features]
            
            # Dự báo tiền điện
            res_money = model.predict([input_row])[0]
            
            # HIỂN THỊ KẾT QUẢ TIỀN ĐIỆN TO RÕ
            st.markdown(f"""
                <div class="result-container">
                    <p class="result-label">Chi phí dự kiến hàng tháng</p>
                    <p class="result-price">{int(round(res_money, -3)):,.0f} <span class="result-unit">VNĐ</span></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Biểu đồ phân bổ điện năng tiêu thụ (giả lập)
            kwh_data = {"Máy lạnh": p_ac_h*22, "Quạt": p_fan_n*p_fan_h*1.5, "Tủ lạnh": 35 if p_fridge else 0, "Bếp": 40 if p_cook else 0, "PC": 20 if p_pc else 0}
            plot_data = {k: v for k, v in kwh_data.items() if v > 0}
            fig = go.Figure(go.Bar(x=list(plot_data.values()), y=list(plot_data.keys()), orientation='h', marker_color='#00f2ff'))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9", height=350, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.balloons()
        else:
            st.info("👈 Nhập đủ thông tin rồi nhấn nút để xem dự báo tiền điện nhé!")
else:
    st.error("Lỗi file dữ liệu!")