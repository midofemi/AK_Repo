import numpy as np


class Perceptron:
  def __init__(self, eta, epochs):
    #This initialize the weights. We used random 3 for (w1,w2, w0). This is because we gonna be using 2 feature X1 and X2
    #But if you look at the note. We always need a bias w0 which make it a total of 3 weight
    #We also multiply it by 1e-4 because we want it to be very small
    self.weights = np.random.randn(3) * 1e-4 # SMALL WEIGHT INIT
    print(f"initial weights before training: \n{self.weights}")
    self.eta = eta # LEARNING RATE
    self.epochs = epochs  


  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights) # z = W * X
    return np.where(z > 0, 1, 0) # CONDITION, IF TRUE, ELSE. If Z > 0 then Z = 1. ELSE 0

  def fit(self, X, y):
    self.X = X
    self.y = y
    #This jut generate an array of bias for you. The bias here is -1
    #-np.ones(len(X), 1) = [-1,-1,-1,-1]
    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))] # CONCATINATION
    print(f"X with bias: \n{X_with_bias}")
    
    #This is our training Looop
    for epoch in range(self.epochs):
      print("--"*10) #Don't be confuse here. This just print many dashes for you time 10. For egs: -----------------------------
      print(f"for epoch: {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(X_with_bias, self.weights) # foward propagation
      print(f"predicted value after forward pass: \n{y_hat}")
      self.error = self.y - y_hat
      print(f"error: \n{self.error}")
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) # backward propagation
      print(f"updated weights after epoch:\n{epoch}/{self.epochs} : \n{self.weights}")
      print("#####"*10)


  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X), 1))]
    return self.activationFunction(X_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total loss: {total_loss}")
    return total_loss