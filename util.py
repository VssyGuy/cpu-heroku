import pandas as pd
    
def label_encoder(df, cols):
    for col in cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    return df

def oneHot_encoder(df, cols):
    for col in cols:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1).drop(columns=[col])
    return df