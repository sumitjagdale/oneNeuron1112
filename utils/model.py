
import numpy as np



class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4 #SMALL WEIGHT INIT
    print(f"Initial weights before the training:{self.weights}")
    self.eta = eta #LEARNING RATE
    self.epochs = epochs
    

  def activationFunction(self, inputs, weights):
    z=np.dot(inputs, weights) # z = w1.x1 + w2x2 + .... => [z= W * X]  
    return np.where(z > 0, 1, 0) # CONDITION , IF TRUE, ELSE
  
  def fit(self, X, y):
    self.X = X
    self.y = y
  
    X_with_bias = np.c_[self.X, -np.ones((len(self.X),1))] #CONCATINATION
    print(f"X with bias: \n{X_with_bias}") 

    for epoch in range (self.epochs):
      print("--"*10)
      print(f"for epoch: {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(X_with_bias, self.weights) # FORWARD PROPAGATION
      print (f"predicted value after forward pass: {y_hat}")
      self.error = self.y - y_hat
      print (f"error: \n{self.error}")
     # updating weights Wn = W0 + Delta(W)
     # Wnew = Wold + eta * error 
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) # BACKWARD PROPAGATION
      print(f"updated weights after epoch: \n {epoch}/{self.epochs} : \n {self.weights}")
      print ("####"*10)

  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X),1))] # X value should not be self.X here
    return self.activationFunction(X_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total_loss: {total_loss}")
    return total_loss