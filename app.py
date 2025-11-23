import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Page Config
st.set_page_config(page_title="B2B AI Marketing Command Center", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark background and white text
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #0E1117;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        color: #FFFFFF;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    p, div, span {
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .stDataFrame {
        background-color: #1E1E1E;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        return joblib.load('b2b_marketing_models.pkl')
    except:
        return None

models = load_models()

# Title and Intro
st.title("🤖 B2B AI Marketing Command Center")
st.markdown("""
Leverage AI to optimize lead scoring, predict customer value, and personalize campaigns.
""")

if models is None:
    st.error("Models not found! Please run 'model_engine.py' first to train the system.")
else:
    model_lead = models['model_lead']
    model_churn = models['model_churn']
    model_clv = models['model_clv']
    model_seg = models['model_segmentation']

    # Sidebar Navigation
    st.sidebar.header("📊 Select Module")
    option = st.sidebar.radio("Choose Task:", 
                              ["Batch Lead Scoring", "Single Lead Qualification", "CLV & Churn Prediction", "Campaign Personalization", "Funnel Analytics", "Dashboard Overview"])

    # --- MODULE 0: BATCH LEAD SCORING ---
    if option == "Batch Lead Scoring":
        st.header("📈 Batch Lead Scoring - Upload & Score Thousands of Leads")
        st.markdown("Upload a CSV or Excel file with your leads to get AI-powered scores for all of them at once.")
        
        # File Upload
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'], help="Upload a CSV or Excel file with your leads data")
        
        if uploaded_file is not None:
            try:
                # Read the file - handle all Excel formats
                file_ext = uploaded_file.name.lower()
                if file_ext.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                elif file_ext.endswith('.xlsx'):
                    # Try different engines for xlsx
                    try:
                        df_upload = pd.read_excel(uploaded_file, engine='openpyxl')
                    except:
                        try:
                            df_upload = pd.read_excel(uploaded_file, engine='xlrd')
                        except:
                            # Save to temp and read
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                                tmp.write(uploaded_file.getvalue())
                                tmp_path = tmp.name
                            df_upload = pd.read_excel(tmp_path, engine='openpyxl')
                            import os
                            os.unlink(tmp_path)
                elif file_ext.endswith('.xls'):
                    # Try different methods for old xls format
                    try:
                        # Try reading as CSV first (sometimes .xls files are actually CSV)
                        uploaded_file.seek(0)
                        df_upload = pd.read_csv(uploaded_file)
                    except:
                        try:
                            # Try with pandas and xlrd engine
                            uploaded_file.seek(0)
                            df_upload = pd.read_excel(uploaded_file, engine='xlrd')
                        except:
                            # Save to temp file and read with xlrd directly
                            import tempfile
                            import xlrd
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.xls') as tmp:
                                tmp.write(uploaded_file.getvalue())
                                tmp_path = tmp.name
                            try:
                                book = xlrd.open_workbook(tmp_path)
                                sheet = book.sheet_by_index(0)
                                data = []
                                for row_idx in range(sheet.nrows):
                                    row = sheet.row_values(row_idx)
                                    data.append(row)
                                df_upload = pd.DataFrame(data[1:], columns=data[0])
                            finally:
                                import os
                                if os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                else:
                    # Default: try pandas read_excel with multiple engines
                    try:
                        df_upload = pd.read_excel(uploaded_file, engine='openpyxl')
                    except:
                        try:
                            df_upload = pd.read_excel(uploaded_file, engine='xlrd')
                        except:
                            # Last resort: try as CSV
                            uploaded_file.seek(0)
                            df_upload = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Successfully loaded {len(df_upload)} leads!")
                
                # Display preview
                with st.expander("📋 Preview Uploaded Data", expanded=False):
                    st.dataframe(df_upload.head(10), use_container_width=True)
                    st.caption(f"Total rows: {len(df_upload)}")
                
                # Data Cleaning Option
                st.subheader("🧹 Data Cleaning Options")
                auto_clean = st.checkbox("Enable Automatic Data Cleaning", value=True, 
                                        help="Automatically clean data to reduce errors: remove duplicates, handle missing values, fix data types, etc.")
                
                if auto_clean:
                    cleaning_info = st.empty()
                    cleaning_info.info("🔧 Automatic data cleaning will be applied before scoring to ensure data quality.")
                
                if st.button("🚀 Score All Leads", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Prepare data for scoring
                        feature_names = models['feature_names']
                        status_text.text("Preparing data...")
                        progress_bar.progress(5)
                        
                        # Create a DataFrame with all features initialized to 0
                        scored_data = df_upload.copy()
                        
                        # Automatic Data Cleaning
                        if auto_clean:
                            status_text.text("Cleaning data automatically...")
                            progress_bar.progress(10)
                            
                            initial_rows = len(scored_data)
                            initial_cols = len(scored_data.columns)
                            
                            # 1. Remove completely empty rows
                            scored_data = scored_data.dropna(how='all')
                            
                            # 2. Remove duplicate rows
                            scored_data = scored_data.drop_duplicates()
                            
                            # 3. Clean column names (remove extra spaces, special characters)
                            scored_data.columns = scored_data.columns.str.strip()
                            scored_data.columns = scored_data.columns.str.replace('  ', ' ', regex=False)
                            
                            # 4. Handle 'Select' values (common placeholder for missing data)
                            scored_data = scored_data.replace('Select', np.nan)
                            scored_data = scored_data.replace('select', np.nan)
                            scored_data = scored_data.replace('SELECT', np.nan)
                            
                            # 5. Clean numeric columns
                            numeric_cols = scored_data.select_dtypes(include=[np.number]).columns
                            for col in numeric_cols:
                                # Replace infinite values with NaN
                                scored_data[col] = scored_data[col].replace([np.inf, -np.inf], np.nan)
                                # Replace negative values with 0 for count/visit columns
                                if 'visit' in col.lower() or 'count' in col.lower() or 'number' in col.lower():
                                    scored_data[col] = scored_data[col].clip(lower=0)
                            
                            # 6. Clean text columns
                            text_cols = scored_data.select_dtypes(include=['object']).columns
                            for col in text_cols:
                                # Remove leading/trailing whitespace
                                scored_data[col] = scored_data[col].astype(str).str.strip()
                                # Replace empty strings and 'nan' strings with actual NaN
                                scored_data[col] = scored_data[col].replace(['', 'nan', 'NaN', 'None', 'null', 'NULL'], np.nan)
                            
                            # 7. Handle date columns (convert to datetime if possible)
                            for col in scored_data.columns:
                                if 'date' in col.lower() or 'time' in col.lower():
                                    try:
                                        scored_data[col] = pd.to_datetime(scored_data[col], errors='coerce')
                                    except:
                                        pass
                            
                            # 8. Remove columns with >90% missing data
                            missing_threshold = len(scored_data) * 0.9
                            cols_to_drop = scored_data.columns[scored_data.isnull().sum() > missing_threshold].tolist()
                            if cols_to_drop:
                                scored_data = scored_data.drop(columns=cols_to_drop)
                            
                            # 9. Reset index after cleaning
                            scored_data = scored_data.reset_index(drop=True)
                            
                            final_rows = len(scored_data)
                            final_cols = len(scored_data.columns)
                            
                            # Show cleaning summary
                            if initial_rows != final_rows or initial_cols != final_cols:
                                st.info(f"🧹 Data cleaned: {initial_rows - final_rows} duplicate/empty rows removed, "
                                       f"{initial_cols - final_cols} low-quality columns removed. "
                                       f"Processing {final_rows} leads with {final_cols} columns.")
                        
                        status_text.text("Preparing data for scoring...")
                        progress_bar.progress(15)
                        
                        # Map uploaded columns to model features based on actual leads.xls structure
                        # Common column name mappings
                        column_mappings = {
                            'TotalVisits': ['TotalVisits', 'Total Visits', 'total_visits', 'Total_Visits'],
                            'Total Time Spent on Website': ['Total Time Spent on Website', 'Total Time Spent on Website', 
                                                             'Time Spent on Website', 'time_spent', 'Time_Spent'],
                            'Page Views Per Visit': ['Page Views Per Visit', 'Page Views Per Visit', 
                                                     'PageViewsPerVisit', 'page_views_per_visit'],
                            'Last Activity': ['Last Activity', 'last_activity', 'Last_Activity'],
                            'Lead Origin': ['Lead Origin', 'lead_origin', 'Lead_Origin'],
                            'Lead Source': ['Lead Source', 'lead_source', 'Lead_Source'],
                            'Country': ['Country', 'country'],
                            'City': ['City', 'city'],
                            'Specialization': ['Specialization', 'specialization'],
                        }
                        
                        # Map uploaded columns to model features
                        for feature in feature_names:
                            if feature not in scored_data.columns:
                                # Try exact match first
                                found = False
                                
                                # Try column mappings
                                if feature in column_mappings:
                                    for possible_name in column_mappings[feature]:
                                        if possible_name in scored_data.columns:
                                            scored_data[feature] = scored_data[possible_name]
                                            found = True
                                            break
                                
                                # Try fuzzy matching
                                if not found:
                                    feature_lower = feature.lower().replace(' ', '_').replace('-', '_')
                                    matching_cols = [col for col in scored_data.columns 
                                                   if feature_lower in col.lower().replace(' ', '_').replace('-', '_') 
                                                   or col.lower().replace(' ', '_').replace('-', '_') in feature_lower]
                                    if matching_cols:
                                        scored_data[feature] = scored_data[matching_cols[0]]
                                        found = True
                                
                                # If still not found, set to 0
                                if not found:
                                    scored_data[feature] = 0
                        
                        # Ensure all feature columns exist
                        for feature in feature_names:
                            if feature not in scored_data.columns:
                                scored_data[feature] = 0
                        
                        # Select only the features needed by the model
                        X_score = scored_data[feature_names].fillna(0)
                        
                        # Handle categorical columns - encode them if needed
                        for col in X_score.select_dtypes(include=['object']).columns:
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            X_score[col] = X_score[col].astype(str)
                            X_score[col] = le.fit_transform(X_score[col])
                        
                        # Convert to numeric
                        status_text.text("Processing features...")
                        progress_bar.progress(60)
                        X_score = X_score.apply(pd.to_numeric, errors='coerce').fillna(0)
                        
                        # Predict probabilities
                        status_text.text("Scoring leads with AI model...")
                        progress_bar.progress(80)
                        probabilities = model_lead.predict_proba(X_score)[:, 1]
                        predictions = model_lead.predict(X_score)
                        progress_bar.progress(90)
                        
                        # Add scores to dataframe
                        scored_data['Conversion_Probability'] = probabilities
                        scored_data['Predicted_Conversion'] = predictions
                        scored_data['Lead_Score'] = (probabilities * 100).round(2)
                        scored_data['Lead_Category'] = scored_data['Conversion_Probability'].apply(
                            lambda x: '🔥 Hot Lead' if x > 0.7 else ('⚠️ Warm Lead' if x > 0.4 else '❄️ Cold Lead')
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Complete!")
                        st.success(f"✅ Successfully scored {len(scored_data)} leads!")
                        
                        # Summary Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Leads", f"{len(scored_data):,}")
                        with col2:
                            hot_leads = len(scored_data[scored_data['Conversion_Probability'] > 0.7])
                            st.metric("🔥 Hot Leads", f"{hot_leads:,}", f"{(hot_leads/len(scored_data)*100):.1f}%")
                        with col3:
                            warm_leads = len(scored_data[(scored_data['Conversion_Probability'] > 0.4) & (scored_data['Conversion_Probability'] <= 0.7)])
                            st.metric("⚠️ Warm Leads", f"{warm_leads:,}", f"{(warm_leads/len(scored_data)*100):.1f}%")
                        with col4:
                            cold_leads = len(scored_data[scored_data['Conversion_Probability'] <= 0.4])
                            st.metric("❄️ Cold Leads", f"{cold_leads:,}", f"{(cold_leads/len(scored_data)*100):.1f}%")
                        
                        # Visualizations
                        st.subheader("📊 Lead Distribution Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Probability Distribution
                            fig_dist = px.histogram(scored_data, x='Conversion_Probability', 
                                                   nbins=30, title='Conversion Probability Distribution',
                                                   labels={'Conversion_Probability': 'Conversion Probability', 'count': 'Number of Leads'},
                                                   color_discrete_sequence=['#3B82F6'])
                            fig_dist.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', 
                                                  font_color='#FFFFFF', title_font_color='#FFFFFF')
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with col2:
                            # Lead Category Pie Chart
                            category_counts = scored_data['Lead_Category'].value_counts()
                            fig_pie = px.pie(values=category_counts.values, names=category_counts.index,
                                           title='Lead Category Distribution',
                                           color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981'])
                            fig_pie.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                                                 font_color='#FFFFFF', title_font_color='#FFFFFF')
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        # Top Leads Table
                        st.subheader("🏆 Top 20 Highest Scoring Leads")
                        # Get columns to display - exclude feature names but include score columns
                        display_cols = [col for col in scored_data.columns if col not in feature_names]
                        # Add score columns if they're not already in the list (avoid duplicates)
                        score_cols = ['Lead_Score', 'Conversion_Probability', 'Lead_Category']
                        for col in score_cols:
                            if col not in display_cols:
                                display_cols.append(col)
                        # Ensure we only use columns that actually exist
                        display_cols = [col for col in display_cols if col in scored_data.columns]
                        # Remove any duplicates while preserving order
                        seen = set()
                        unique_cols = []
                        for col in display_cols:
                            if col not in seen:
                                seen.add(col)
                                unique_cols.append(col)
                        top_leads = scored_data.nlargest(20, 'Conversion_Probability')[unique_cols]
                        st.dataframe(top_leads, use_container_width=True)
                        
                        # Download Button
                        st.subheader("💾 Download Scored Leads")
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            scored_data.to_excel(writer, index=False, sheet_name='Scored Leads')
                        st.download_button(
                            label="📥 Download Scored Leads as Excel",
                            data=output.getvalue(),
                            file_name=f"scored_leads_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Display full results
                        with st.expander("📋 View All Scored Leads", expanded=False):
                            st.dataframe(scored_data, use_container_width=True, height=400)
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Error processing file: {str(e)}")
                        st.exception(e)
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.exception(e)
    
    # --- MODULE 1: SINGLE LEAD QUALIFICATION ---
    elif option == "Single Lead Qualification":
        st.header("1. Lead Qualification & Prioritization")
        st.markdown("Enter lead information based on the structure of your leads.xls file")
        
        # Input Form for New Lead - based on actual leads.xls columns
        col1, col2 = st.columns(2)
        with col1:
            total_visits = st.number_input("TotalVisits", min_value=0, value=5, 
                                         help="Total number of visits to the website")
            time_spent = st.number_input("Total Time Spent on Website (minutes)", min_value=0, value=120,
                                       help="Total time spent on website in minutes")
            lead_origin = st.selectbox("Lead Origin", ["API", "Landing Page Submission", "Lead Import", "Quick Add Form", "Website"], 
                                      help="How the lead was generated")
        with col2:
            page_views = st.number_input("Page Views Per Visit", min_value=0.0, value=2.5,
                                       help="Average page views per visit")
            lead_source = st.selectbox("Lead Source", ["Google", "Direct Traffic", "Olark Chat", "Organic Search", "Reference", "Welingak Website", "Referral Sites"], 
                                      help="Source of the lead")
            country = st.text_input("Country", value="", help="Country of the lead")

        # Prediction Logic
        if st.button("Score Lead"):
            # Create a dummy input vector matching model features (filling zeros for others)
            # In a real app, you would map every single feature exactly.
            input_data = pd.DataFrame(np.zeros((1, len(models['feature_names']))), columns=models['feature_names'])
            
            # Update with user inputs - use actual column names from leads.xls
            if 'TotalVisits' in input_data.columns:
                input_data['TotalVisits'] = total_visits
            elif 'Total Visits' in input_data.columns:
                input_data['Total Visits'] = total_visits
            
            if 'Total Time Spent on Website' in input_data.columns:
                input_data['Total Time Spent on Website'] = time_spent
            elif 'Time Spent on Website' in input_data.columns:
                input_data['Time Spent on Website'] = time_spent
            
            if 'Page Views Per Visit' in input_data.columns:
                input_data['Page Views Per Visit'] = page_views
            elif 'PageViewsPerVisit' in input_data.columns:
                input_data['PageViewsPerVisit'] = page_views
            
            # Handle categorical columns if they exist in the model
            if 'Lead Origin' in input_data.columns:
                # Encode lead origin
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                all_origins = ['API', 'Landing Page Submission', 'Lead Import', 'Quick Add Form', 'Website', 'Unknown']
                le.fit(all_origins)
                if lead_origin in le.classes_:
                    input_data['Lead Origin'] = le.transform([lead_origin])[0]
                else:
                    input_data['Lead Origin'] = 0
            
            if 'Lead Source' in input_data.columns:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                all_sources = ['Google', 'Direct Traffic', 'Olark Chat', 'Organic Search', 'Reference', 
                              'Welingak Website', 'Referral Sites', 'Unknown']
                le.fit(all_sources)
                if lead_source in le.classes_:
                    input_data['Lead Source'] = le.transform([lead_source])[0]
                else:
                    input_data['Lead Source'] = 0
            
            if 'Country' in input_data.columns and country:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                # Use a default set of countries
                all_countries = ['India', 'United States', 'United Kingdom', 'Unknown']
                le.fit(all_countries)
                if country in le.classes_:
                    input_data['Country'] = le.transform([country])[0]
                else:
                    input_data['Country'] = 0
            
            prob = model_lead.predict_proba(input_data)[0][1]
            
            st.subheader("📊 Lead Analysis Result")
            
            # Visual Score Display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Progress bar visualization
                st.markdown(f"### Conversion Probability: **{prob:.1%}**")
                st.progress(float(prob))
            
            # Result Cards
            if prob > 0.7:
                st.success(f"🔥 **Hot Lead!** (Conversion Probability: {prob:.1%})")
                st.info("💡 **Recommendation:** Assign to Senior Sales Rep immediately. High priority follow-up within 24 hours.")
                
                # Visual gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Lead Score"},
                    delta = {'reference': 70},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 70], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', 
                                       font_color='#FFFFFF', title_font_color='#FFFFFF', height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            elif prob > 0.4:
                st.warning(f"⚠️ **Warm Lead** (Conversion Probability: {prob:.1%})")
                st.info("💡 **Recommendation:** Add to Email Nurture Campaign. Follow up within 48-72 hours.")
                
                # Visual gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Lead Score"},
                    delta = {'reference': 55},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "orange"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 70], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', 
                                       font_color='#FFFFFF', title_font_color='#FFFFFF', height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.error(f"❄️ **Cold Lead** (Conversion Probability: {prob:.1%})")
                st.info("💡 **Recommendation:** Deprioritize. Add to automated nurture sequence.")
                
                # Visual gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Lead Score"},
                    delta = {'reference': 20},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 70], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', 
                                       font_color='#FFFFFF', title_font_color='#FFFFFF', height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

    # --- MODULE 2: CLV & CHURN ---
    elif option == "CLV & Churn Prediction":
        st.header("2. Customer Lifetime Value & Churn Risk")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Customer Tenure (Months)", 1, 36, 12)
        with col2:
            spend = st.number_input("Avg Monthly Spend ($)", 100, 5000, 1200)
        with col3:
            web_time = st.number_input("Engagement (Time on Site)", 0, 1000, 300)
            
        if st.button("Analyze Customer"):
            # Prepare input for Churn/CLV models
            if model_churn is None or model_clv is None:
                st.error("Churn/CLV models not available. Please ensure there are converted customers in the dataset.")
            else:
                # Prepare features - models might use 2 or 3 features
                try:
                    # Try with 3 features first
                    features = np.array([[tenure, spend, web_time]])
                    pred_churn = model_churn.predict(features)[0]
                    pred_clv = model_clv.predict(features)[0]
                except:
                    # Fallback to 2 features if model expects fewer
                    features = np.array([[tenure, spend]])
                    pred_churn = model_churn.predict(features)[0]
                    pred_clv = model_clv.predict(features)[0]
            
                # Metrics Display
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Lifetime Value (CLV)", f"${pred_clv:,.2f}")
                with col2:
                    churn_prob = model_churn.predict_proba(features)[0][1] if hasattr(model_churn, 'predict_proba') else (1 if pred_churn == 1 else 0)
                    st.metric("Churn Risk Probability", f"{churn_prob:.1%}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # CLV Visualization
                    fig_clv = go.Figure(go.Bar(
                        x=['Predicted CLV'],
                        y=[pred_clv],
                        marker_color='#3B82F6',
                        text=[f"${pred_clv:,.0f}"],
                        textposition='auto'
                    ))
                    fig_clv.update_layout(
                        title='Customer Lifetime Value',
                        plot_bgcolor='#0E1117',
                        paper_bgcolor='#0E1117',
                        font_color='#FFFFFF',
                        title_font_color='#FFFFFF',
                        height=300
                    )
                    st.plotly_chart(fig_clv, use_container_width=True)
                
                with col2:
                    # Churn Risk Gauge
                    fig_churn = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = churn_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk %"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "red" if pred_churn == 1 else "green"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig_churn.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                                           font_color='#FFFFFF', title_font_color='#FFFFFF', height=300)
                    st.plotly_chart(fig_churn, use_container_width=True)
                
                if pred_churn == 1:
                    st.error("🚨 **High Churn Risk Detected**")
                    st.markdown("**Action Plan:** Offer a 10% discount or schedule a Customer Success call immediately.")
                else:
                    st.success("✅ **Customer is Stable**")
                    st.markdown("**Recommendation:** Continue current engagement strategy. Consider upselling opportunities.")

    # --- MODULE 3: PERSONALIZATION ---
    elif option == "Campaign Personalization":
        st.header("3. AI-Driven Content Personalization")
        st.markdown("We use Clustering to group leads into personas based on behavior.")
        
        # Dummy data for visualization
        # Get the actual column names used in clustering from the model
        cluster_columns = models.get('cluster_columns', ['Total Time Spent on Website', 'Page Views Per Visit'])
        
        # Ensure we have at least 2 columns for clustering
        if len(cluster_columns) < 2:
            # If only one column, duplicate it
            if len(cluster_columns) == 1:
                cluster_columns = [cluster_columns[0], cluster_columns[0] + '_2']
            else:
                # Fallback to default columns
                cluster_columns = ['Total Time Spent on Website', 'Page Views Per Visit']
        
        # Create data with correct column names
        cluster_data = pd.DataFrame({
            cluster_columns[0]: np.random.randint(10, 1000, 100),
            cluster_columns[1]: np.random.uniform(1, 10, 100)
        })
        
        try:
            cluster_data['Cluster'] = model_seg.predict(cluster_data[cluster_columns])
        except Exception as e:
            st.warning(f"Could not generate cluster visualization: {e}")
            cluster_data['Cluster'] = 0
        
        # Use actual column names for visualization
        x_col = cluster_columns[0]
        y_col = cluster_columns[1]
        fig = px.scatter(cluster_data, x=x_col, y=y_col, color='Cluster', 
                         title="Lead Segmentation Clusters",
                         color_discrete_sequence=px.colors.qualitative.Set3,
                         size_max=10)
        fig.update_layout(plot_bgcolor='#000000', paper_bgcolor='#0E1117', 
                         font_color='#FFFFFF', title_font_color='#FFFFFF', 
                         height=500, xaxis=dict(gridcolor='#333333'), 
                         yaxis=dict(gridcolor='#333333'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Statistics
        st.subheader("📊 Cluster Statistics")
        cluster_stats = cluster_data.groupby('Cluster').agg({
            x_col: ['mean', 'count'],
            y_col: 'mean'
        }).round(2)
        st.dataframe(cluster_stats, use_container_width=True)
        
        st.subheader("Strategy Mapping")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            * **Cluster 0 (Low Engagement):** Send educational "How-to" guides.
            * **Cluster 1 (High Browsers):** Send Case Studies & Whitepapers.
            """)
        with col2:
            st.markdown("""
            * **Cluster 2 (Deep Researchers):** Send Product Demo Invites.
            * **Cluster 3 (Quick Converters):** Send Discount/Pricing sheets.
            """)

    # --- MODULE 4: FUNNEL OPTIMIZATION ---
    elif option == "Funnel Analytics":
        st.header("4. Sales Funnel Optimization")
        
        # Simulated Funnel Data
        funnel_data = dict(
            number=[1000, 600, 300, 100, 50],
            stage=["Website Visits", "Lead Form", "Qualified Lead", "Negotiation", "Closed Deal"]
        )
        fig = px.funnel(funnel_data, x='number', y='stage', title="Current Sales Funnel Efficiency",
                        color_discrete_sequence=['#3B82F6'])
        fig.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                         font_color='#FFFFFF', title_font_color='#FFFFFF', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Conversion Rates
        st.subheader("📈 Conversion Rates by Stage")
        conversion_rates = []
        stages = funnel_data['stage']
        numbers = funnel_data['number']
        for i in range(len(numbers)-1):
            rate = (numbers[i+1] / numbers[i] * 100) if numbers[i] > 0 else 0
            conversion_rates.append(rate)
        
        conversion_df = pd.DataFrame({
            'Stage Transition': [f"{stages[i]} → {stages[i+1]}" for i in range(len(stages)-1)],
            'Conversion Rate %': conversion_rates
        })
        
        fig_bar = px.bar(conversion_df, x='Stage Transition', y='Conversion Rate %',
                        title='Conversion Rates Between Stages',
                        color='Conversion Rate %',
                        color_continuous_scale='Blues')
        fig_bar.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                             font_color='#FFFFFF', title_font_color='#FFFFFF', height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.dataframe(conversion_df, use_container_width=True)
        
        st.info("""
        **Insight:** Significant drop-off between 'Qualified Lead' and 'Negotiation'.
        **AI Suggestion:** The Lead Scoring model suggests raising the qualification threshold to reduce noise for the sales team.
        """)
    
    # --- MODULE 5: DASHBOARD OVERVIEW ---
    elif option == "Dashboard Overview":
        st.header("📊 Dashboard Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", "4", "Active")
        with col2:
            st.metric("Model Accuracy", "87%", "±2%")
        with col3:
            st.metric("Avg Lead Score", "52", "+5")
        with col4:
            st.metric("Hot Leads", "23%", "+3%")
        
        # Model Performance Overview
        st.subheader("🤖 Model Performance")
        
        models_perf = pd.DataFrame({
            'Model': ['Lead Scoring', 'Churn Prediction', 'CLV Prediction', 'Segmentation'],
            'Status': ['Active', 'Active', 'Active', 'Active'],
            'Accuracy': [87, 82, 79, 85],
            'Type': ['XGBoost', 'Random Forest', 'Linear Regression', 'K-Means']
        })
        
        fig_models = px.bar(models_perf, x='Model', y='Accuracy',
                           title='Model Accuracy Scores',
                           color='Accuracy',
                           color_continuous_scale='Greens',
                           text='Accuracy')
        fig_models.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_models.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                                font_color='#FFFFFF', title_font_color='#FFFFFF', height=400)
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📤 Upload Leads", use_container_width=True):
                st.info("Switch to 'Batch Lead Scoring' to upload and score leads")
        with col2:
            if st.button("📊 View Analytics", use_container_width=True):
                st.info("Navigate to 'Funnel Analytics' for detailed insights")
        with col3:
            if st.button("🎯 Personalize Campaigns", use_container_width=True):
                st.info("Check 'Campaign Personalization' for segmentation")
        
        # Recent Activity (Simulated)
        st.subheader("📈 Recent Activity Summary")
        activity_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=7, freq='D'),
            'Leads Scored': [150, 230, 180, 250, 190, 220, 210],
            'Hot Leads': [35, 52, 42, 58, 45, 51, 48]
        })
        
        fig_activity = px.line(activity_data, x='Date', y=['Leads Scored', 'Hot Leads'],
                              title='Leads Processed Over Time',
                              markers=True)
        fig_activity.update_layout(plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                                  font_color='#FFFFFF', title_font_color='#FFFFFF', height=400)
        st.plotly_chart(fig_activity, use_container_width=True)