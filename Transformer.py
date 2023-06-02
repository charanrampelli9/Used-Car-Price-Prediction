from ced19i041_hcl_project import linearRegression,X_train_df,standardScaler
import pandas as pd
def Transformer(X_test):
    print(X_test.columns)
    X_test = pd.get_dummies(X_test,columns = ["Brand", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first = True)
    missing_cols = set(X_train_df.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train_df.columns]
    X_test = standardScaler.transform(X_test)
    y_pred = linearRegression.predict(X_test)
    return y_pred