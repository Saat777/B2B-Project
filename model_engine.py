import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import xgboost as xgb
import joblib

# --- 1. Data Loading & Synthetic Enhancement ---
def load_and_prep_data(filepath):
    # Load original dataset (Assumed generic name, replace with actual file name if different)
    # For this example, we simulate loading if file doesn't exist or load actual
    df = None
    file_attempts = [
        ('leads.xls', 'csv'),  # Try as CSV first (file might be misnamed)
        ('leads.xls', 'xlrd'),
        ('leads.xlsx', 'openpyxl'),
        ('Leads.csv', None),  # CSV doesn't need engine
        ('leads.csv', None)
    ]
    
    for filename, engine in file_attempts:
        try:
            if engine == 'csv':
                # Try reading .xls file as CSV (sometimes files are misnamed)
                try:
                    df = pd.read_csv(filename)
                    print(f"Successfully loaded {filename} as CSV")
                    break
                except Exception as e:
                    continue
            elif engine == 'xlrd':
                # For old .xls files, use xlrd directly to avoid pandas version conflicts
                try:
                    import xlrd
                    book = xlrd.open_workbook(filename)
                    sheet = book.sheet_by_index(0)
                    # Convert to list of lists
                    data = []
                    for row_idx in range(sheet.nrows):
                        row = sheet.row_values(row_idx)
                        data.append(row)
                    # First row as header
                    df = pd.DataFrame(data[1:], columns=data[0])
                    print(f"Successfully loaded {filename} using xlrd direct read")
                    break
                except Exception as e:
                    continue
            elif engine:
                df = pd.read_excel(filename, engine=engine)
                print(f"Successfully loaded {filename}")
                break
            else:
                df = pd.read_csv(filename)
                print(f"Successfully loaded {filename}")
                break
        except FileNotFoundError:
            continue
        except Exception as e:
            continue
    
    if df is None:
        print("Dataset not found. Please ensure 'leads.xls', 'leads.xlsx', or 'Leads.csv' is in the directory.")
        return None

    # Basic Cleaning: 'Select' often means null in this dataset
    df = df.replace('Select', np.nan)
    
    # Drop columns with > 40% missing data (standard practice)
    limit = len(df) * 0.6
    df = df.dropna(thresh=limit, axis=1)

    # --- SYNTHETIC DATA GENERATION FOR GOALS 2 & 3 (CLV/Churn) ---
    # The original dataset only has 'Converted'. We need post-purchase data.
    # We generate this ONLY for converted leads to simulate a customer base.
    
    np.random.seed(42)
    
    # Simulate 'Tenure' (Months) and 'Monthly_Spend'
    df['Tenure_Months'] = np.random.randint(1, 36, size=len(df))
    df['Monthly_Spend'] = np.random.uniform(100, 5000, size=len(df))
    
    # Simulate 'Churn' (1 = Churned, 0 = Retained)
    # Logic: Lower tenure and spend = higher churn probability
    df['Churn_Risk'] = (df['Tenure_Months'] < 6) & (df['Monthly_Spend'] < 1000)
    df['Churn'] = df['Churn_Risk'].apply(lambda x: 1 if x and np.random.random() > 0.3 else 0)
    
    # Calculate CLV
    df['CLV'] = df['Monthly_Spend'] * df['Tenure_Months']
    
    return df

# --- 2. Preprocessing Pipeline ---
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', LabelEncoder()) # We will use OneHot in a real scenario, but LabelEncoder for simplicity here
    ])
    
    # Note: For production, OneHotEncoder is safer. Here we perform manual encoding for simplicity
    # to fit everything in one learnable script context.
    
    return numeric_features, categorical_features

# --- 3. Model Training ---
def train_models(df):
    print("Preprocessing Data...")
    
    # Target 1: Lead Qualification (Original Dataset Target)
    # Try to find the converted column with different possible names
    target_lead = None
    for col_name in ['Converted', 'converted', 'CONVERTED', 'Conversion', 'conversion']:
        if col_name in df.columns:
            target_lead = col_name
            break
    
    if target_lead is None:
        raise ValueError("Could not find 'Converted' column in dataset. Please ensure the dataset has a conversion target column.")
    
    # Features for Lead Scoring
    drop_cols = ['Prospect ID', 'Lead Number', 'Prospect_ID', 'Lead_Number', target_lead, 'Churn', 'CLV', 'Churn_Risk']
    # Ensure we only drop columns that actually exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X_lead = df.drop(columns=drop_cols)
    y_lead = df[target_lead]
    
    # Handle Categorical Encoding manually for simplicity in this example
    for col in X_lead.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_lead[col] = X_lead[col].astype(str)
        X_lead[col] = le.fit_transform(X_lead[col])
        
    # Fill NaN
    X_lead = X_lead.fillna(0)

    # A. Lead Scoring Model (XGBoost)
    print("Training Lead Scoring Model...")
    model_lead = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_lead.fit(X_lead, y_lead)

    # B. Churn Prediction Model (Random Forest)
    # We only train this on Converted customers
    print("Training Churn Model...")
    customer_df = df[df[target_lead] == 1].copy()
    if len(customer_df) > 0:
        y_churn = customer_df['Churn']
        # Find available time-related columns
        time_cols = [col for col in ['Total Time Spent on Website', 'Time Spent on Website', 'Total_Time_Spent_on_Website'] if col in customer_df.columns]
        if time_cols:
            X_churn = customer_df[['Tenure_Months', 'Monthly_Spend', time_cols[0]]]
        else:
            # Use only available columns
            X_churn = customer_df[['Tenure_Months', 'Monthly_Spend']]
        model_churn = RandomForestClassifier()
        model_churn.fit(X_churn, y_churn)
    else:
        model_churn = None
        print("Not enough converted data for Churn model.")

    # C. CLV Prediction (Linear Regression)
    print("Training CLV Model...")
    if len(customer_df) > 0:
        y_clv = customer_df['CLV']
        # Find available time-related columns
        time_cols = [col for col in ['Total Time Spent on Website', 'Time Spent on Website', 'Total_Time_Spent_on_Website'] if col in customer_df.columns]
        if time_cols:
            X_clv = customer_df[['Tenure_Months', 'Monthly_Spend', time_cols[0]]]
        else:
            # Use only available columns
            X_clv = customer_df[['Tenure_Months', 'Monthly_Spend']]
        model_clv = LinearRegression()
        model_clv.fit(X_clv, y_clv)
    else:
        model_clv = None

    # D. Personalization (Clustering)
    print("Training Segmentation Model...")
    # We cluster based on behavioral metrics
    # Find available columns for clustering
    cluster_cols = []
    for col in ['Total Time Spent on Website', 'Time Spent on Website', 'Total_Time_Spent_on_Website']:
        if col in X_lead.columns:
            cluster_cols.append(col)
            break
    for col in ['Page Views Per Visit', 'Page_Views_Per_Visit', 'Page Views']:
        if col in X_lead.columns:
            cluster_cols.append(col)
            break
    
    if len(cluster_cols) >= 2:
        X_cluster = X_lead[cluster_cols]
    elif len(cluster_cols) == 1:
        # If only one column available, duplicate it for clustering
        X_cluster = X_lead[[cluster_cols[0], cluster_cols[0]]]
    else:
        # Fallback to first two numeric columns
        numeric_cols = X_lead.select_dtypes(include=['int64', 'float64']).columns[:2]
        if len(numeric_cols) < 2:
            numeric_cols = list(numeric_cols) + [numeric_cols[0]] if len(numeric_cols) > 0 else []
        X_cluster = X_lead[numeric_cols[:2]]
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_cluster)

    return model_lead, model_churn, model_clv, kmeans, X_lead.columns, list(X_cluster.columns)

if __name__ == "__main__":
    # 1. Load
    df = load_and_prep_data('Leads.csv')
    
    if df is not None:
        # 2. Train
        model_lead, model_churn, model_clv, kmeans, feature_names, cluster_columns = train_models(df)
        
        # 3. Save Models
        data_to_save = {
            'model_lead': model_lead,
            'model_churn': model_churn,
            'model_clv': model_clv,
            'model_segmentation': kmeans,
            'feature_names': feature_names,
            'cluster_columns': cluster_columns
        }
        joblib.dump(data_to_save, 'b2b_marketing_models.pkl')
        print("All models trained and saved to 'b2b_marketing_models.pkl'")