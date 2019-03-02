import numpy as np
import matplotlib.pyplot as plt


def split(X):
    size = len(X)
    rate = int(size/5)
    test_x = X[:rate]
    train_x = X[rate:]
    return test_x, train_x

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

accuracyies = []

def minibatch_gradient_descent(X, y, learning_rate=0.01, iterations=10, batch_size=32):

    weights = np.zeros(X.shape[1])

    m = len(y)
    cost_history = np.zeros(iterations)
    weights_history = np.zeros((iterations, 2))

    for i in range(iterations):
        cost_per_iteration = .0

        for j in range(0, m, batch_size):
            X_inner = X[i: i + batch_size]
            y_inner = y[i: i + batch_size]

            scores = np.dot(X_inner, weights)
            predictions = sigmoid(scores)

            # Update weights with gradient
            output_error_signal = y_inner - predictions
            gradient = np.dot(X_inner.T, output_error_signal)
            weights += learning_rate * gradient

            ll = log_likelihood(X_inner, y_inner, weights)
            cost_per_iteration += ll


        final_scores = np.dot(X, weights)
        preds = np.round(sigmoid(final_scores))
        accuracy = format((preds == y).sum().astype(float) / len(preds))
        a = float(accuracy)
        accuracyies.append('{:06.5f}'.format(a))
        cost_history[i] = cost_per_iteration

    return weights, cost_history, weights_history


datapath='ionosphere.data.csv'

df = np.loadtxt(datapath,delimiter=',',dtype=str)#data okundu

np.random.seed(30)
np.random.shuffle(df)#veri random olarak karıştırıldı

A, y = df[:, :-1], df[:, -1]#target ve attributelar ayrıldı

X = np.empty(shape=[len(A),len(A[0])], dtype=float)
for i in range(len(A)):
    for j in range(len(A[0])):
        X[i][j]=float(A[i][j])#attributelar numpy float arrayine çevrildi

Y = []
for i in range(len(y)):#target değerleri numeric arraya çevirildi
    if(y[i]=='g'):
        Y.append(1)
    else :
        Y.append(0)

test_x , train_x = split(X)#test ve train dataları %20 lik oranla ayrıldı
test_y , train_y = split(Y)

weights,cost_history, theta_history = minibatch_gradient_descent(train_x,train_y,0.009,250,32)#mini-batch gradient modeli çağırıldı
plt.plot(accuracyies)#accurysi ekranda gösterildi
plt.show()

final_scores = np.dot(test_x, weights)
preds = np.round(sigmoid(final_scores))
accuracy = format((preds == test_y).sum().astype(float) / len(preds))
a = float(accuracy)
print(a)