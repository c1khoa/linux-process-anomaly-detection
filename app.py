import json
import joblib
import os
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

MODEL_PATH = "demo/best_model.pkl"
SCALER_PATH = "demo/z_scaler.pkl"
FEATURE_COLS_PATH = "demo/feature_columns.json"
TEST_FILE_PATH = "demo/test.csv"
SCALE_COLS = ["argsNum", "returnValue"]

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_COLS_PATH, "r") as f:
            feature_cols = json.load(f)
        return model, scaler, feature_cols
    except Exception as e:
        return None, None, str(e)

model, scaler, FEATURE_COLS = load_artifacts()

st.set_page_config(page_title="Phát Hiện Bất Thường Kiểu Mạng", layout="wide")

st.markdown("""
    <style>
    .scanner-text { font-family: 'Courier New', Courier, monospace; font-size: 14px; }
    .anomaly-list { background-color: #ffeded; border-radius: 5px; padding: 10px; border: 1px solid #ff4b4b; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center; }
    .stButton button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("Phát hiện bất thường trong tiến trình của hệ điều hành Linux")
st.caption("Mô phỏng bắt tín hiệu tiến trình theo thời gian thực")

if "anomaly_history" not in st.session_state:
    st.session_state.anomaly_history = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "all_processes" not in st.session_state:
    st.session_state.all_processes = []

with st.sidebar:
    st.header("Thông số cài đặt")
    speed = st.slider("Tốc độ quét (giây/tiến trình)", 0.1, 2.0, 0.5)
    if st.button("Bắt đầu" if not st.session_state.is_running else "Tiếp tục"):
            st.session_state.is_running = True
            st.rerun()
        
    if st.button("Tạm dừng"):
            st.session_state.is_running = False
            st.rerun()
    
    if st.button("Làm mới tiến trình"):
        st.session_state.anomaly_history = []
        st.session_state.current_index = 0
        st.session_state.is_running = False
        st.session_state.all_processes = []
        st.rerun()
    
    st.divider()
    st.subheader("Thống kê tổng quan")
    total_scanned = len(st.session_state.all_processes)
    total_anomalies = len(st.session_state.anomaly_history)
    anomaly_rate = (total_anomalies / total_scanned * 100) if total_scanned > 0 else 0
    
    st.metric("Tổng tiến trình đã quét", total_scanned)
    st.metric("Phát hiện anomaly", total_anomalies)
    st.metric("Tỷ lệ anomaly", f"{anomaly_rate:.1f}%")
    
    st.divider()
    uploaded_file = st.file_uploader("Hoặc tải file test mới", type=["csv"])

source_file = uploaded_file if uploaded_file else (TEST_FILE_PATH if os.path.exists(TEST_FILE_PATH) else None)

if source_file:
    df_test = pd.read_csv(source_file)

    tab1, tab2, tab3 = st.tabs(["Giám Sát Trực Tiếp", "Lịch sử", "Chi Tiết"])
    
    with tab1:
        col_live, col_anomaly = st.columns([1.5, 1])

        with col_live:
            st.subheader("Luồng Tiến Trình Trực Tiếp")
            progress_bar = st.progress(0)
            stream_placeholder = st.empty()

        with col_anomaly:
            st.subheader("Các tiến trình bát thường gần đây")
            anomaly_placeholder = st.empty()

    with tab2:
        st.subheader("Thống kê")
        stats_placeholder = st.empty()
        chart_placeholder = st.empty()
    
    with tab3:
        st.subheader("Chi tiết")
        if st.session_state.anomaly_history:
            selected_anomaly = st.selectbox(
                "Chọn bất thường để xem chi tiết:",
                options=range(len(st.session_state.anomaly_history)),
                format_func=lambda x: f"ID: {st.session_state.anomaly_history[x]['id']} - {st.session_state.anomaly_history[x]['name']}"
            )
            
            if selected_anomaly is not None:
                item = st.session_state.anomaly_history[selected_anomaly]
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.info(f"**ID Tiến Trình:** {item['id']}")
                    st.info(f"**Tên Tiến Trình:** {item['name']}")
                with col_info2:
                    st.info(f"**Thời điểm phát hiện:** {item.get('timestamp', 'N/A')}")
                
                st.divider()
                st.write("### Đặc Trưng")
                
                features_df = pd.DataFrame([item['raw_data']]).T
                features_df.columns = ['Giá Trị']
                features_df.index.name = 'Đặc Trưng'
                st.dataframe(features_df, use_container_width=True)
        else:
            st.info("Chưa có bất thường nào được phát hiện. Hãy bắt đầu giám sát!")

    if st.session_state.is_running:
        logs = []
        
        for index in range(st.session_state.current_index, len(df_test)):
            if not st.session_state.is_running:
                break
                
            row = df_test.iloc[index]
            raw_features = row[FEATURE_COLS].to_dict()

            X_df = pd.DataFrame([raw_features])
            X_scaled = X_df.copy()
            X_scaled[SCALE_COLS] = scaler.transform(X_scaled[SCALE_COLS])
            
            pred = model.predict(X_scaled[FEATURE_COLS])[0]

            process_info = {
                "id": index,
                "name": row.get('name', f'Proc_{index}'),
                "prediction": pred,
                "timestamp": time.strftime("%H:%M:%S")
            }
            st.session_state.all_processes.append(process_info)

            status_icon = "⚪" if pred == 0 else "🔴"
            log_entry = f"{status_icon} ID: {index} | Name: {row.get('name', 'Proc_'+str(index))} | Threads: {row.get('threadId', 0)} | Time: {process_info['timestamp']}"
            logs.insert(0, log_entry)
            stream_placeholder.code("\n".join(logs[:15]))

            if pred == 1:
                anomaly_item = {
                    "id": index,
                    "name": row.get('name', f"Process_{index}"),
                    "raw_data": raw_features,
                    "timestamp": process_info['timestamp']
                }
                st.session_state.anomaly_history.append(anomaly_item)

            with anomaly_placeholder.container():
                for item in reversed(st.session_state.anomaly_history[-5:]):
                    with st.expander(f"🔴 ID: {item['id']} - {item['name']} ({item['timestamp']})"):
                        st.write("**Đặc trưng:**")
                        st.json(item['raw_data'])
            
            with stats_placeholder.container():
                if st.session_state.anomaly_history:
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("Tổng Bất Thường", len(st.session_state.anomaly_history))
                    with col_s2:
                        recent_anomalies = sum(1 for p in st.session_state.all_processes[-20:] if p['prediction'] == 1)
                        st.metric("Bất Thường (20 gần nhất)", recent_anomalies)
                    with col_s3:
                        st.metric("Tiến trình bình thường", len(st.session_state.all_processes) - len(st.session_state.anomaly_history))
            
            with chart_placeholder:
                if len(st.session_state.all_processes) > 1:
                    recent_data = st.session_state.all_processes[-50:]
                    df_chart = pd.DataFrame(recent_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_chart.index,
                        y=df_chart['prediction'],
                        mode='lines+markers',
                        name='Prediction',
                        line=dict(color='red'),
                        marker=dict(size=8)
                    ))
                    fig.update_layout(
                        title="Anomaly Detection Timeline (Last 50 processes)",
                        xaxis_title="Process Index",
                        yaxis_title="Prediction (0=Normal, 1=Anomaly)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            progress = (index + 1) / len(df_test)
            progress_bar.progress(progress)
            
            st.session_state.current_index = index + 1
            time.sleep(speed)
        
        if st.session_state.current_index >= len(df_test):
            st.success("🏁 Hoàn thành quét file test.")
            st.session_state.is_running = False
            st.balloons()

    elif st.session_state.current_index == 0:
        stream_placeholder.info("Nhấn 'Bắt đầu' để chạy giám sát.")
    else:
        stream_placeholder.warning("Đã tạm dừng. Nhấn 'Tiếp tục' để chạy tiếp.")

else:
    st.error("❌ Không tìm thấy file test.csv")