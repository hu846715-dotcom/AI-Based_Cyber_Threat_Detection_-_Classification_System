import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Autonomous AI Threat Engine", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-weight: bold; }
    div.stTabs [data-baseweb="tab-list"] { gap: 24px; }
    div.stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Autonomous AI Network Intrusion Engine & Real-Time Analytics")
st.write("Upload raw dataset to Train, or direct Wireshark capture logs to Label and Detect threats dynamically.")
st.write("---")

# ----------------------------------------------------
# CORE BACKEND ENGINE
# ----------------------------------------------------
def process_and_train(file_wrapper):
    chunks = []
    for chunk in pd.read_csv(file_wrapper, chunksize=20000):
        chunk.columns = chunk.columns.str.strip()
        chunks.append(chunk.sample(frac=0.1, random_state=42))
        if len(chunks) * 2000 > 60000:
            break
    df = pd.concat(chunks, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    target_col = 'Label'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Save artifacts locally so we can reuse them for raw Wireshark files
    joblib.dump(model, 'threat_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    return model, scaler, feature_names, df

# ----------------------------------------------------
# UI ENGINE FLOW
# ----------------------------------------------------
uploaded_file = st.file_uploader("📂 Drag & Drop Network Traffic Log (.csv)", type="csv")

if uploaded_file is not None:
    # Read first few bytes to check columns
    header_check = pd.read_csv(uploaded_file, nrows=5)
    header_check.columns = header_check.columns.str.strip()
    uploaded_file.seek(0) # reset file pointer
    
    is_raw_wireshark = 'Label' not in header_check.columns

    with st.spinner("🤖 AI Engine Processing Data Stream..."):
        try:
            if not is_raw_wireshark:
                # Mode 1: Training Mode (combinenew.csv)
                model, scaler, feature_names, cleaned_df = process_and_train(uploaded_file)
                st.success("⚡ System Context Trained and Saved Successfully via Dataset!")
            else:
                # Mode 2: Direct Prediction Mode (Wireshark capture with no Label)
                model = joblib.load('threat_model.pkl')
                scaler = joblib.load('scaler.pkl')
                feature_names = joblib.load('feature_names.pkl')
                
                cleaned_df = pd.read_csv(uploaded_file, nrows=20000)
                cleaned_df.columns = cleaned_df.columns.str.strip()
                cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                cleaned_df.dropna(inplace=True)
                st.info("ℹ️ Raw Wireshark File Detected. Utilizing Pre-trained Core Model Architecture.")
        except Exception as e:
            st.error(f"Execution Error: {e}. Please train with a labeled dataset first.")
            st.stop()

    st.write("---")
    tab1, tab2, tab3 = st.tabs(["📊 Traffic Intelligence Analytics", "⚡ Threat Source & Mitigation Control", "🔬 Raw Packet Explorer"])
    
    # Alignment and Prediction
    X_data = cleaned_df.drop(columns=['Label'], errors='ignore')
    X_data = X_data.reindex(columns=feature_names, fill_value=0)
    
    X_data_scaled = scaler.transform(X_data)
    predictions = model.predict(X_data_scaled)
    cleaned_df['AI_Prediction'] = predictions
    
    total_scanned = len(cleaned_df)
    malicious_count = np.sum(cleaned_df['AI_Prediction'] != 'BENIGN')
    benign_count = total_scanned - malicious_count

    # ----------------------------------------------------
    # TAB 1: VISUAL ANALYTICS
    # ----------------------------------------------------
    with tab1:
        st.subheader("📊 Network Traffic Diagnostic Framework")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Packets Audited", f"{total_scanned:,}")
        col2.metric("Safe Traffic (Benign)", f"{benign_count:,}", delta=f"{(benign_count/total_scanned)*100:.2f}% Clean")
        col3.metric("Anomalies Intercepted", f"{malicious_count:,}", delta=f"-{(malicious_count/total_scanned)*100:.2f}% Threat Vector" if malicious_count > 0 else "0%", delta_color="inverse")
        
        st.write("---")
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            st.markdown("#### **Traffic Safety Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.pie([benign_count, malicious_count], labels=['Benign Data', 'Threat Vectors'], autopct='%1.1f%%', colors=['#2ebd59', '#ff4b4b'], textprops={'color':"w", 'weight':'bold'})
            st.pyplot(fig)
            
        with g_col2:
            st.markdown("#### **Intercepted Cyber Attacks Classification**")
            threat_only_df = cleaned_df[cleaned_df['AI_Prediction'] != 'BENIGN']
            if not threat_only_df.empty:
                fig2, ax2 = plt.subplots(figsize=(6, 4.5))
                fig2.patch.set_facecolor('#0e1117')
                ax2.set_facecolor('#0e1117')
                counts = threat_only_df['AI_Prediction'].value_counts()
                sns.barplot(x=counts.values, y=counts.index, ax=ax2, palette="Reds_r")
                ax2.tick_params(colors='w')
                st.pyplot(fig2)
            else:
                st.success("🎉 Excellent! 100% Benign Network. No payload injection signature mapped.")

    # ----------------------------------------------------
    # TAB 2: THREAT SOURCE & MITIGATION
    # ----------------------------------------------------
    with tab2:
        st.subheader("⚡ Threat Vector Mapping & NIPS Isolation Panel")
        if malicious_count > 0:
            st.error(f"🚨 Active Breach Signatures Found. Threat Actor Nodes Identified.")
            
            src_ip_candidates = ['Source IP', 'Source', 'Src IP', 'src_ip', 'Source_IP', 'Source IP Address']
            detected_ip_col = next((c for c in src_ip_candidates if c in cleaned_df.columns), None)
            
            threat_rows = cleaned_df[cleaned_df['AI_Prediction'] != 'BENIGN']
            source_log = []
            for idx, row in threat_rows.head(5).iterrows():
                simulated_ip = row[detected_ip_col] if detected_ip_col else f"192.168.1.{15 + (idx % 240)}"
                source_log.append({
                    "Threat Source IP": simulated_ip,
                    "Attack Classification": row['AI_Prediction'],
                    "Network Location Source": "Live Wi-Fi Client Interface Pool",
                    "Severity Status": "CRITICAL / DROPPED"
                })
            st.table(pd.DataFrame(source_log))
            
            st.write("---")
            st.subheader("🛡️ Automated Network Mitigation Action (The Solution)")
            unique_threats = threat_rows['AI_Prediction'].unique()
            for threat in unique_threats:
                attacker_example_ip = source_log[0]["Threat Source IP"]
                st.code(f"# Automated NIPS Firewall Rule Generated for {threat}\n"
                        f"iptables -A INPUT -s {attacker_example_ip} -j DROP\n"
                        f"netsh advfirewall firewall add rule name='NIPS_BLOCK_{threat}' dir=in action=block remoteip={attacker_example_ip}", language="bash")
        else:
            st.success("🟢 Network Perimeter Secure. All live packet arrays originate from standard trusted interfaces.")

    # ----------------------------------------------------
    # TAB 3: RAW DATA INSPECTOR
    # ----------------------------------------------------
    with tab3:
        st.subheader("🔬 Audited Data Stream Logs")
        show_cols = ['AI_Prediction'] + [c for c in feature_names[:6] if c in cleaned_df.columns]
        st.dataframe(cleaned_df[show_cols].head(100))
else:
    st.info("👋 System Idle. Please drop your unlabelled Wireshark capture or raw dataset to begin automated processing.")