import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Problem 2
def main2(data):
    diamonds =data

    # Feature and target matrices
    X = diamonds[['High Temp (°F)', 'Low Temp (°F)', 'Precipitation']]
    y = diamonds[['Total']]

    # Training and testing split, with 25% of the data reserved as the test set
    X = X.to_numpy()
    total_true = y.to_numpy()

    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)


    [X_train, trn_mean, trn_std] = normalize_train(X_train,3)
    X_test = normalize_test(X_test, trn_mean, trn_std)


    lmbda = np.logspace(-1, 2, num=51)  # fill in

    MODEL = []
    MSE = []
    for l in lmbda:
        model = train_model(X_train, y_train, l)

        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)

        MODEL.append(model)
        MSE.append(mse)

    plt.plot(lmbda, MSE)
    plt.xlabel('Lambda (Regularization Parameter)', fontsize=17)
    plt.ylabel('Mean Squared Error', fontsize=17)
    plt.title("Regularization of features High Temp,Low Temp,Precipitation and Total")

    plt.show()

    # Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))  # fill in
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))


    price_pred = np.zeros(np.shape(X[:,0]))

    for x in range(len(X)):
        newValues = np.array(X[x])
        newNorm = (newValues-trn_mean)/trn_std
        price_pred[x] = np.dot(model_best.coef_, newNorm) + model_best.intercept_
    print("predicted traffic by normalizing data",end=' ')
    print(price_pred)
    r2Value = r2_score(total_true,price_pred)

    print("R^2 Value : ",(r2Value))


    return model_best

# main 3
def main3(data):
    diamonds =data

    # Feature and target matrices
    X = diamonds[['Total']]
    y = diamonds[['Precipitation']]

    # Training and testing split, with 25% of the data reserved as the test set
    X = X.to_numpy()
    total_true = y.to_numpy()

    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    regr = LinearRegression(fit_intercept=True)
    regr.fit(X_train,y_train)
    y_predTest = regr.predict(X_test)
    plt.scatter(X_train, y_train, color='black', label='Train data points')
    plt.scatter(X_test, y_test, color='red', label='Test data points')
    plt.plot(X_test, y_predTest, color='blue', linewidth=1, label='Model')
    plt.scatter(X_test, y_predTest, marker='x', color='red', linewidth=3,)
    plt.legend()
    plt.show()

    [X_train, trn_mean, trn_std] = normalize_train(X_train,1)
    X_test = normalize_test(X_test, trn_mean, trn_std)


    lmbda = np.logspace(-1, 2, num=51)  # fill in

    MODEL = []
    MSE = []
    for l in lmbda:
        model = train_model(X_train, y_train, l)

        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)

        MODEL.append(model)
        MSE.append(mse)

    plt.plot(lmbda, MSE)
    plt.xlabel('Lambda (Regularization Parameter)', fontsize=17)
    plt.ylabel('Mean Squared Error', fontsize=17)
    plt.title("Regularization of features Precipitation and Total")
    plt.show()

    # Find best value of lmbda in terms of MSE
    ind = MSE.index(min(MSE))  # fill in
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))


    price_pred = np.zeros(np.shape(X[:,0]))

    for x in range(len(X)):
        newValues = np.array(X[x])
        newNorm = (newValues-trn_mean)/trn_std
        price_pred[x] = np.dot(model_best.coef_, newNorm) + model_best.intercept_
    print("predicted precipitation by normalizing data",end=' ')
    print(price_pred)
    r2Value = r2_score(total_true,price_pred)

    print("R^2 Value : ",(r2Value))


    return model_best


def normalize_train(X_train,number):
    # fill in
    X_norm = np.zeros(np.shape(X_train))
    meanData = np.zeros(np.shape(X_train[0]))
    standardDev = np.zeros(np.shape(X_train[0]))
    for x in range(number):
        X_norm[:, x] = (X_train[:, x] - np.mean(X_train[:, x])) / np.std(X_train[:, x])
        meanData[x] = np.mean(X_train[:, x])
        standardDev[x] = np.std(X_train[:, x])

    X = X_norm

    return X, meanData, standardDev


def normalize_test(X_test, trn_mean, trn_std):
    X_norm = np.empty(np.shape(X_test))
    for x in range(len(X_test[0])):
        X_norm[:, x] = (X_test[:, x] - trn_mean[x]) / trn_std[x]

    X = X_norm

    # fill in

    return X


def train_model(X, y, l):
    model = linear_model.Ridge(alpha=l, fit_intercept=True)
    model.fit(X, y)

    # fill in

    return model

def error(X, y, model):
    y = y.to_numpy()
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)

    # Fill in

    return mse



# data input
def inputs():
    # read data using csv and replacing comma value
    data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    data = data.replace(',', '', regex=True)

    # Removing values not having 0.
    for i in range(len(data)):
        if data.loc[i,'Precipitation'] == 'T':
            data.loc[i, 'Precipitation'] = 0

    # Converting data to their respective types
    data['Precipitation'] = data['Precipitation'].astype(float)
    data['Brooklyn Bridge'] = data['Brooklyn Bridge'].astype(int)
    data['Manhattan Bridge'] = data['Manhattan Bridge'].astype(int)
    data['Williamsburg Bridge'] = data['Williamsburg Bridge'].astype(int)
    data['Queensboro Bridge'] = data['Queensboro Bridge'].astype(int)
    data['Total']=data['Total'].astype(int)

    return data


def sensorsPlace(data):

    bridgeA=[]
    bridgeB=[]
    bridgeC=[]
    bridgeD=[]

    for i in range(len(data)):
        bridgeA.append(data.loc[i, 'Brooklyn Bridge'] / data.loc[i, 'Total'])
        bridgeB.append(data.loc[i, 'Manhattan Bridge'] / data.loc[i, 'Total'])
        bridgeC.append(data.loc[i, 'Williamsburg Bridge'] / data.loc[i, 'Total'])
        bridgeD.append(data.loc[i, 'Queensboro Bridge'] / data.loc[i, 'Total'])

    sumA = sum(bridgeA)
    sumB = sum(bridgeB)
    sumC = sum(bridgeC)
    sumD = sum(bridgeD)

    sumAll = sumA+sumB+sumC+sumD

    sumA = sumA / sumAll
    sumB = sumB / sumAll
    sumC = sumC / sumAll
    sumD = sumD / sumAll

    y = np.array([sumA,sumB,sumC,sumD])
    mylabels = ["Brooklyn Bridge", "Manhattan Bridge", "Williamsburg Bridge", "Queensboro Bridge"]
    plt.pie(y,labels = mylabels,autopct='%0.f%%')
    plt.show()

    dict={

        "Brooklyn Bridge" : sumA,
        "Manhattan Bridge": sumB,
        "Williamsburg Bridge": sumC,
        "Queensboro Bridge": sumD
    }
    return dict


if __name__ == '__main__':
    data = inputs()
    print("Place the sensors for bridges with higher data:")
    print('The probability of a randomly selected bicyclist riding on one of the four bridges is listed:')
    listBridge = sensorsPlace(data)
    print(listBridge)
    print("----------------------------------------------------------------------------")
    plt.scatter(data['High Temp (°F)'],data['Total'])
    plt.title("Correlation between High Temp and Total")
    plt.xlabel("High Temp (°F)")
    plt.ylabel("Total Traffic")
    plt.show()


    plt.scatter(data['Low Temp (°F)'], data['Total'])
    plt.title("Correlation between Low Temp (°F) and Total")
    plt.xlabel("Low Temp (°F)")
    plt.ylabel("Total Traffic")
    plt.show()


    plt.scatter(data['Precipitation'], data['Total'])
    plt.title("Correlation between Precipitation and Total")
    plt.xlabel("Precipitation")
    plt.ylabel("Total Traffic")
    plt.show()

    #['High Temp (°F)', 'Low Temp (°F)', 'Precipitation']
    main2(data)
    print("----------------------------------------------------------------------------")
    main3(data)
    plt.scatter(data['Total'],data['Precipitation'])
    plt.title("Correlation between Total and Precipitation")
    plt.ylabel("Precipitation")
    plt.xlabel("Total Traffic")
    plt.show()


