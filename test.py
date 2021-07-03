# create feedforward neural network in pytorch 
# that moment when copilot knows ai better than you 

class Feedforward(): 
    def __init__(self, input_size, hidden_size, output_size): 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b1 = np.random.randn(self.hidden_size)
        self.b2 = np.random.randn(self.output_size)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1) + self.b1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2) + self.b2
        yHat = self.sigmoid(self.z3)

        return yHat
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
    
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)