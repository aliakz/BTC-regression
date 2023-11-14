#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import yfinance as yf
#Loading the Dataset

bitcoin = pd.read_csv("BTC-USD.csv")
# or u can use yf librarie to get  dataset   

#Exploring the Dataset
print("data sets size :",bitcoin.shape)
print(bitcoin.head())
print(bitcoin.info)
print(bitcoin.index)
Columns=[]
for x in bitcoin.columns:
    Columns.append(x)
print(" Columns :", Columns)

plt.figure(figsize = (12, 7))
plt.plot(bitcoin["Date"], bitcoin["Open"], color='goldenrod', lw=2)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)
plt.show()


# Preprocessing the Data

print(bitcoin.isnull().sum())
bitcoin.dropna(inplace=True)
bitcoin["Benefit"]=bitcoin["Close"]-bitcoin['Open']
required_features = ['Open', 'High', 'Low', 'Volume' ]
output_label = 'Close'

x_train, x_test, y_train, y_test = train_test_split(
bitcoin[required_features],
bitcoin[output_label],
test_size = 0.3
)

#Creating the Model
model = LinearRegression()
model.fit(x_train, y_train)

score=model.score(x_test, y_test)
print("model score is :",score*100)

#predict X_test 

prediction = model.predict(x_test)

print('mse :' ,mean_squared_error(y_test,prediction))
days =np.arange(1,len(prediction)+1)
plt.scatter(days,prediction,color='red')
days =np.arange(1,len(y_test)+1)
plt.scatter(days,y_test)
plt.show()

#createing  a future dataset by shifting the original data and pridect future useing model 

future_set = bitcoin.shift(periods=30).tail(30)
prediction = model.predict(future_set[required_features])

plt.figure(figsize = (15, 7))
plt.plot(bitcoin["Date"][-400:-60], bitcoin["Open"][-400:-60], color='blue', lw=2)
plt.plot(future_set["Date"], prediction, color='deeppink', lw=3)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel(rotation = -45)
plt.xlabel("Time", rotation = -45,size=20)
plt.legend()
plt.ylabel("$ Price", size=20)

plt.show()