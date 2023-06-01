import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)



if __name__ == "__main__":
    X_train, y_train = load_pickle("./output/train.pkl")
    X_test, y_test = load_pickle("./output/test.pkl")

    rf = RandomForestRegressor(
        max_depth=15,
        min_samples_leaf=4,
        min_samples_split=2,
        n_estimators=34,
        n_jobs=-1,
        random_state=42
            )
    rf.fit(X_train, y_train)

    predictions_test = rf.predict(X_test)
    print(mean_squared_error(y_test, predictions_test, squared=False))


    
