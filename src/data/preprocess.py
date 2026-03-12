import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):

    df = df.drop_duplicates()

    if "target" in df.columns:
        label = "target"
    elif "output" in df.columns:
        label = "output"
    else:
        label = df.columns[-1]

    X = df.drop(label, axis=1)
    y = df[label]

    # convert multiclass -> binary
    y = (y > 0).astype(int)

    X = pd.get_dummies(X)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X