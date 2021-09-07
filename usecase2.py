import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

#loading
provider_1 = pd.read_csv("C:\\Users\\ilker\\Desktop\\Datathon\\usecase2\\Provider1_Usecase2.csv", sep=";")
provider_2 = pd.read_csv("C:\\Users\\ilker\\Desktop\\Datathon\\usecase2\\Provider2_Usecase2.csv", sep=";")
provider_3 = pd.read_csv("C:\\Users\\ilker\\Desktop\\Datathon\\usecase2\\Provider3_Usecase2.csv", sep=";")
actual = pd.read_csv("C:\\Users\\ilker\\Desktop\\Datathon\\usecase2\\ActualGeneration_Usecase2_v2.csv", sep=";")

provider_1.isna().sum()
provider_2.isna().sum()
provider_3.isna().sum()

#date time

from datetime import datetime


provider_1["Datekey"] = provider_1["Datekey"].astype(str)
provider_2["Datekey"] = provider_1["Datekey"].astype(str)
provider_3["Datekey"] = provider_1["Datekey"].astype(str)

date_arr = []
for h in range(len(provider_1)):
    date = provider_1["Datekey"][h]
    date = date[2:]
    date = date[:2] + "/" + date[2:4] + "/" + date[4:] + " " + provider_1["Hour"][h]
    date = datetime.strptime(date, '%y/%m/%d %H:%M')
    date_arr.append(date)

type(date_arr[0])
provider_1["Date-Hour"] = date_arr

date_arr = []
for h in range(len(provider_2)):
    date = provider_2["Datekey"][h]
    date = date[2:]
    date = date[:2] + "/" + date[2:4] + "/" + date[4:] + " " + provider_2["Hour"][h]
    date = datetime.strptime(date, '%y/%m/%d %H:%M')
    date_arr.append(date)

provider_2["Date-Hour"] = date_arr

date_arr = []
for h in range(len(provider_3)):
    date = provider_3["Datekey"][h]
    date = date[2:]
    date = date[:2] + "/" + date[2:4] + "/" + date[4:] + " " + provider_3["Hour"][h]
    date = datetime.strptime(date, '%y/%m/%d %H:%M')
    date_arr.append(date)
provider_3["Date-Hour"] = date_arr

provider_2["Year"] = provider_2["Date-Hour"].dt.year
provider_2["Month"] = provider_2["Date-Hour"].dt.month
provider_2["Week"] = provider_2["Date-Hour"].dt.week
provider_2["Day"] = provider_2["Date-Hour"].dt.day
provider_2["DayofWeek"] = provider_2["Date-Hour"].dt.dayofweek
provider_2["Hour"] = provider_2["Date-Hour"].dt.hour

provider_1["Year"] = provider_1["Date-Hour"].dt.year
provider_1["Month"] = provider_1["Date-Hour"].dt.month
provider_1["Week"] = provider_1["Date-Hour"].dt.week
provider_1["Day"] = provider_1["Date-Hour"].dt.day
provider_1["DayofWeek"] = provider_1["Date-Hour"].dt.dayofweek
provider_1["Hour"] = provider_1["Date-Hour"].dt.hour

provider_3["Year"] = provider_3["Date-Hour"].dt.year
provider_3["Month"] = provider_3["Date-Hour"].dt.month
provider_3["Week"] = provider_3["Date-Hour"].dt.week
provider_3["Day"] = provider_3["Date-Hour"].dt.day
provider_3["DayofWeek"] = provider_3["Date-Hour"].dt.dayofweek
provider_3["Hour"] = provider_3["Date-Hour"].dt.hour

#NA values
from sklearn.impute import KNNImputer


imputer = KNNImputer(n_neighbors=24)
p2 = imputer.fit_transform(provider_2.drop(["Datekey", "ProviderId", "Date-Hour"], axis=1)) #actualu koy

p2 = pd.DataFrame(p2, 
             columns=['Hour', 
                      'WindSpeed',
                      "WindDirection",
                      "Temperature",
                      "Pressure",
                      "Humidity",
                      "PowerWOAvailability",
                      "Year",
                      "Month",
                      "Week",
                      "Day",
                      "DayofWeek"])

p2["Datekey"] = provider_2["Datekey"]
p2["ProviderId"] = provider_2["ProviderId"]
p2["Date-Hour"] = provider_2["Date-Hour"]

p3 = imputer.fit_transform(provider_3.drop(["Datekey", "ProviderId", "Date-Hour"], axis=1)) #actualu koy
p3 = pd.DataFrame(p3, 
             columns=['Hour', 
                      'WindSpeed',
                      "WindDirection",
                      "Temperature",
                      "Pressure",
                      "Humidity",
                      "PowerWOAvailability",
                      "Year",
                      "Month",
                      "Week",
                      "Day",
                      "DayofWeek"])

p3["Datekey"] = provider_3["Datekey"]
p3["ProviderId"] = provider_3["ProviderId"]
p3["Date-Hour"] = provider_3["Date-Hour"]

p1 = provider_1.copy()

cor = p2.corr()

#Wind Direction Encoding

import math 

sine_arr = []
cos_arr = []
for g in range(len(p1)):
    rad = math.radians(p1["WindDirection"][g])
    sine_arr.append(math.sin(rad))
    cos_arr.append(math.cos(rad))
p1["Direction_Sin"] = sine_arr
p1["Direction_Cos"] = cos_arr

sine_arr = []
cos_arr = []
for g in range(len(p2)):
    rad = math.radians(p2["WindDirection"][g])
    sine_arr.append(math.sin(rad))
    cos_arr.append(math.cos(rad)) 
p2["Direction_Sin"] = sine_arr
p2["Direction_Cos"] = cos_arr

sine_arr = []
cos_arr = []
for g in range(len(p3)):
    rad = math.radians(p3["WindDirection"][g])
    sine_arr.append(math.sin(rad))
    cos_arr.append(math.cos(rad))
p3["Direction_Sin"] = sine_arr
p3["Direction_Cos"] = cos_arr

p2["Hour"] = p2["Hour"].astype(int)
p3["Hour"] = p3["Hour"].astype(int)
p1["WindDirection"] = p1["WindDirection"].astype(int)

# sin-cos hour
p1["Hour_sin"] = np.sin(2 * np.pi * p1["Hour"]/23.0)
p2["Hour_sin"] = np.sin(2 * np.pi * p2["Hour"]/23.0)
p3["Hour_sin"] = np.sin(2 * np.pi * p3["Hour"]/23.0)

p1["Hour_cos"] = np.cos(2 * np.pi * p1["Hour"]/23.0)
p2["Hour_cos"] = np.cos(2 * np.pi * p2["Hour"]/23.0)
p3["Hour_cos"] = np.cos(2 * np.pi * p3["Hour"]/23.0)


type(p1["Hour_sin"][0])
type(p1["Direction_Sin"][0])

#One-Hot Encoding -> Direction, Year, DayofWeek, Month


p1["Humidity"] = p1["Humidity"]/100
p1 = p1.drop(["Datekey", "ProviderId", "Date-Hour", "Week", "Day", "WindDirection"], axis=1)
p2 = p2.drop(["Datekey", "ProviderId", "Date-Hour", "Year", "Month", "Week", "Day", "DayofWeek", "WindDirection"], axis=1)
p3 = p3.drop(["Datekey", "ProviderId", "Date-Hour", "Year", "Month", "Week", "Day", "DayofWeek", "WindDirection"], axis=1)

p2.columns = ['Hour', 'WindSpeed2', 'Temperature2', 'Pressure2',
       'Humidity2', 'PowerWOAvailability2', 'Direction_Sin2', 'Direction_Cos2',
       'Hour_sin2', 'Hour_cos2']
p3.columns = ['Hour', 'WindSpeed3', 'Temperature3', 'Pressure3',
       'Humidity3', 'PowerWOAvailability3', 'Direction_Sin3', 'Direction_Cos3',
       'Hour_sin3', 'Hour_cos3']

n = pd.concat([p1, p2], axis=1, join="inner")
n = pd.concat([n, p3], axis=1, join="inner")

n_onehot = pd.get_dummies(n, columns = ["Year", "DayofWeek", "Month"], prefix= ["Year", "DayofWeek", "Month"])
n_onehot = n_onehot.drop(["Year_2021", "DayofWeek_6", "Month_12"], axis=1)

n_onehot = n_onehot.drop(["Hour"], axis=1)

new= pd.concat([n_onehot, actual], axis=1, join="inner")
new.replace([np.inf, -np.inf], np.nan, inplace=True)
new = new.dropna()

y = new["PowerMWh"]
X = new.drop(["PCTimeStamp", "PowerMWh"], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

cat = CatBoostRegressor(iterations=200,
                          learning_rate=0.1,
                          depth=16)

cat.fit(X_train, y_train)

cat_pred = cat.predict(X_test)

mean_squared_error(y_test, cat_pred, squared=False)