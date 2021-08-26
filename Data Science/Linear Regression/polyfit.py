import numpy as np
import matplotlib.pyplot as plt


# Return fitted model parameters to the dataset at datapath for each choice in degrees.
# Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
# Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
# coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []

    file = open(datapath, 'r')
    data = file.readlines()

    x_data = []
    y_data = []
    for line in data:
        [i, j] = line.split()
        x_data.append(float(i))
        y_data.append(float(j))

    for n in degrees:
        X = feature_matrix(x_data,n)
        B = least_squares(X,y_data)
        paramFits.append(B)

    return paramFits


# Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.
# Input: x as a list of the independent variable samples, and d as an integer.
# Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
# for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):
    # fill in
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    X = [[i ** k for k in range(d, -1, -1)] for i in x]

    return X


# Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
# Input: X as a list of features for each sample, and y as a list of target variable samples.
# Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    # fill in
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.

    B = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))

    return B


if __name__ == '__main__':
    datapath = 'poly.txt'
    # degrees = [2, 4]

    file = open(datapath, 'r')
    data = file.readlines()
    x_data = []
    y_data = []
    for lines in data:
        [i, j] = lines.split()
        x_data.append(float(i))
        y_data.append(float(j))

    degrees = [1, 2, 3, 4, 5]
    paramFits = main(datapath, degrees)
    print(paramFits)

    plt.scatter(x_data, y_data, color='black', label='data')

    x_data.sort()
    for param in paramFits:
        d = len(param) - 1
        X = feature_matrix(x_data, d)
        X = np.array(X)
        y_predicted = np.dot(X, param)
        plt.plot(x_data, y_predicted, label='d = ' + str(d))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=10, loc='upper left')
    plt.show()


    degrees = [2, 4]

    paramFits = main(datapath, degrees)
    print(paramFits)

