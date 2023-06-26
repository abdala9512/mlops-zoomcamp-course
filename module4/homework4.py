#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import pickle

with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


# In[4]:


categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


df = read_data("../data/yellow_tripdata_2022-02.parquet")

dicts = df[categorical].to_dict(orient="records")
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[5]:


import numpy as np

np.std(y_pred)


# In[9]:


df["ride_id"] = f"{2022:04d}/{2:02d}_" + df.index.astype("str")
df["prediction"] = y_pred
df[["ride_id", "prediction"]]


# In[10]:


df_result = df[["ride_id", "prediction"]]
df_result.to_parquet(
    "output_predictions.parquet", engine="pyarrow", compression=None, index=False
)


# In[ ]:
