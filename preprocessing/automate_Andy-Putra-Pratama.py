import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_and_preprocess(df):
    df = df.drop(columns=['customer_id'])
    
    cat_cols = ['contract_type', 'has_internet']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
    
    num_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print("Automated preprocessing successful.")

if __name__ == "__main__":
    df_raw = load_data('../customer_churn_raw.csv')
    df_clean = clean_and_preprocess(df_raw)
    save_data(df_clean, 'namadataset_preprocessing/customer_churn_preprocessing.csv')
