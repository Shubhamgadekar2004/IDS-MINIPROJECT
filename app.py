import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        with open('best_ids_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        return model, scaler, label_encoders, target_encoder
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first.")
        return None, None, None, None

# Preprocess data function
def preprocess_data(df, label_encoders, scaler):
    df_processed = df.copy()
    
    # Handle categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    for col in categorical_cols:
        if col in df_processed.columns and col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            unseen_mask = ~df_processed[col].isin(le.classes_)
            if unseen_mask.any():
                most_frequent = le.classes_[0]
                df_processed.loc[unseen_mask, col] = most_frequent
            df_processed[col] = le.transform(df_processed[col])
    
    # Scale features
    df_scaled = scaler.transform(df_processed)
    
    return df_scaled

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Network Intrusion Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Cybersecurity Threat Detection")
    st.markdown("---")
    
    # Load models
    model, scaler, label_encoders, target_encoder = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-checked.png", width=100)
        st.title("📊 Control Panel")
        st.markdown("---")
        
        page = st.radio("Navigation", ["🏠 Home", "📤 Single Prediction", "📁 Batch Prediction", "📈 Analytics"])
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This IDS uses machine learning to detect network intrusions and anomalies in real-time.")
        
        st.markdown("### 🎯 Attack Types")
        st.markdown("""
        - **Normal**: Legitimate traffic
        - **Anomaly**: Suspicious activity
        """)
    
    # Home Page
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Status", "✅ Active", "Ready")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Type", type(model).__name__, "Trained")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Classes", len(target_encoder.classes_), "Detection Types")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("## 🚀 Quick Start Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Single Prediction")
            st.write("Upload a single network traffic record for instant analysis")
            st.code("""
            1. Go to Single Prediction page
            2. Upload CSV with one record
            3. Get instant results
            """)
        
        with col2:
            st.markdown("### 📁 Batch Prediction")
            st.write("Analyze multiple network records at once")
            st.code("""
            1. Go to Batch Prediction page
            2. Upload CSV with multiple records
            3. Download results
            """)
        
        st.markdown("---")
        st.markdown("### 📋 Required Data Format")
        st.write("Your CSV should contain these 41 features:")
        
        expected_features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
                           'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
                           'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                           'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                           'num_access_files', 'num_outbound_cmds', 'is_host_login',
                           'is_guest_login', 'count', 'srv_count', 'serror_rate',
                           'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                           'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                           'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                           'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                           'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                           'dst_host_srv_rerror_rate']
        
        st.dataframe(pd.DataFrame({'Feature Name': expected_features}), height=300)
    
    # Single Prediction Page
    elif page == "📤 Single Prediction":
        st.markdown("## 📤 Single Record Prediction")
        st.write("Upload a CSV file with a single network traffic record for instant threat analysis.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="single")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Show uploaded data
                with st.expander("📋 View Uploaded Data"):
                    st.dataframe(df)
                
                if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
                    with st.spinner("Analyzing network traffic..."):
                        # Preprocess
                        X_processed = preprocess_data(df, label_encoders, scaler)
                        
                        # Predict
                        prediction = model.predict(X_processed)
                        prediction_proba = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
                        predicted_class = target_encoder.inverse_transform(prediction)[0]
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 🎯 Detection Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if predicted_class == 'normal':
                                st.success("### ✅ NORMAL TRAFFIC")
                                st.markdown("**Status:** Safe")
                            else:
                                st.error("### ⚠️ ANOMALY DETECTED")
                                st.markdown("**Status:** Potential Threat")
                        
                        with col2:
                            st.metric("Classification", predicted_class.upper())
                            if prediction_proba is not None:
                                confidence = np.max(prediction_proba[0]) * 100
                                st.metric("Confidence", f"{confidence:.2f}%")
                        
                        # Probability distribution
                        if prediction_proba is not None:
                            st.markdown("### 📊 Confidence Distribution")
                            prob_df = pd.DataFrame({
                                'Class': target_encoder.classes_,
                                'Probability': prediction_proba[0] * 100
                            })
                            
                            fig = px.bar(prob_df, x='Class', y='Probability', 
                                       color='Probability',
                                       color_continuous_scale='RdYlGn_r',
                                       title='Prediction Probabilities')
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance (if available)
                        if hasattr(model, 'feature_importances_'):
                            with st.expander("🔬 Top Contributing Features"):
                                feature_importance = pd.DataFrame({
                                    'Feature': df.columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False).head(10)
                                
                                fig = px.bar(feature_importance, x='Importance', y='Feature',
                                           orientation='h', title='Top 10 Feature Importances')
                                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Batch Prediction Page
    elif page == "📁 Batch Prediction":
        st.markdown("## 📁 Batch Prediction")
        st.write("Upload a CSV file with multiple network traffic records for batch analysis.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="batch")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Analyzing {len(df)} records...")
                
                # Show sample data
                with st.expander("📋 View Sample Data (First 5 rows)"):
                    st.dataframe(df.head())
                
                if st.button("🔍 Analyze All Records", type="primary", use_container_width=True):
                    with st.spinner("Processing batch predictions..."):
                        # Preprocess
                        X_processed = preprocess_data(df, label_encoders, scaler)
                        
                        # Predict
                        predictions = model.predict(X_processed)
                        predictions_proba = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
                        predicted_classes = target_encoder.inverse_transform(predictions)
                        
                        # Add predictions to dataframe
                        results_df = df.copy()
                        results_df['predicted_class'] = predicted_classes
                        if predictions_proba is not None:
                            results_df['confidence'] = np.max(predictions_proba, axis=1) * 100
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Batch Analysis Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Records", len(results_df))
                        with col2:
                            normal_count = (predicted_classes == 'normal').sum()
                            st.metric("Normal Traffic", normal_count)
                        with col3:
                            anomaly_count = (predicted_classes == 'anomaly').sum()
                            st.metric("Anomalies", anomaly_count)
                        with col4:
                            if len(results_df) > 0:
                                anomaly_rate = (anomaly_count / len(results_df)) * 100
                                st.metric("Threat Rate", f"{anomaly_rate:.1f}%")
                        
                        # Visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart
                            pred_counts = pd.Series(predicted_classes).value_counts()
                            fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                       title='Traffic Classification Distribution',
                                       color_discrete_sequence=['#00CC96', '#EF553B'])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Bar chart
                            fig = px.bar(x=pred_counts.index, y=pred_counts.values,
                                       title='Classification Counts',
                                       labels={'x': 'Class', 'y': 'Count'},
                                       color=pred_counts.index,
                                       color_discrete_sequence=['#00CC96', '#EF553B'])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(results_df, height=400)
                        
                        # Download results
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f'ids_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Analytics Page
    elif page == "📈 Analytics":
        st.markdown("## 📈 Model Analytics")
        
        st.info("📊 Upload test data with predictions to view detailed analytics")
        
        uploaded_file = st.file_uploader("Upload predictions CSV", type=['csv'], key="analytics")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'predicted_class' in df.columns:
                    st.success("✅ Data loaded successfully!")
                    
                    # Overall statistics
                    st.markdown("### 📊 Overall Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Analyzed", len(df))
                    with col2:
                        normal = (df['predicted_class'] == 'normal').sum()
                        st.metric("Normal Traffic", normal)
                    with col3:
                        anomalies = (df['predicted_class'] == 'anomaly').sum()
                        st.metric("Anomalies Detected", anomalies)
                    
                    # Time series (if duration available)
                    if 'duration' in df.columns:
                        st.markdown("### ⏱️ Traffic Analysis Over Duration")
                        df_sorted = df.sort_values('duration')
                        fig = px.scatter(df_sorted, x='duration', y=df_sorted.index,
                                       color='predicted_class',
                                       title='Traffic Classification Over Time',
                                       color_discrete_map={'normal': '#00CC96', 'anomaly': '#EF553B'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Protocol analysis
                    if 'protocol_type' in df.columns:
                        st.markdown("### 🌐 Protocol Distribution")
                        protocol_stats = df.groupby(['protocol_type', 'predicted_class']).size().reset_index(name='count')
                        fig = px.bar(protocol_stats, x='protocol_type', y='count',
                                   color='predicted_class', barmode='group',
                                   title='Threats by Protocol Type',
                                   color_discrete_map={'normal': '#00CC96', 'anomaly': '#EF553B'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("⚠️ File must contain 'predicted_class' column")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()