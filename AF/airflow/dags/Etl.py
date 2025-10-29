import pandas as pd
import numpy as np
import os
class ExtractLoadTransformSave:
    def __init__(self):
        pass

    def data_loader(self, **kwargs):
        import pandas as pd
        path = "/opt/airflow/data/flights.csv"
        df = pd.read_csv(path, skip_blank_lines=True)
        return df

    def data_transformation(self, **kwargs):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        path = "/opt/airflow/data/flights.csv"
        df = pd.read_csv(path)
        df.drop(columns=['travelCode', 'userCode', 'time', 'date'], inplace=True)

        le = LabelEncoder()
        df['from'] = le.fit_transform(df['from'])
        df['to'] = le.fit_transform(df['to'])
        df['flightType'] = le.fit_transform(df['flightType'])
        df['agency'] = le.fit_transform(df['agency'])

        df.to_csv("/opt/airflow/data/flights_encode.csv", index=False)

    def model_training(self, **kwargs):
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        import pickle

        path = "/opt/airflow/data/flights_encode.csv"
        df = pd.read_csv(path)
        X = df.drop(columns=['price'])
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=29)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("MSR:", mean_squared_error(y_test, y_pred))
        print("R2 score:", r2_score(y_test, y_pred))

        with open("/opt/airflow/data/model.pkl", "wb") as f:
            pickle.dump(model, f)
