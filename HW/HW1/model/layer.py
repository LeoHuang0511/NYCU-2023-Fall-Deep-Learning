import numpy as np




class Layer:
    inputs: np.ndarray
    params: dict[str, np.ndarray] = {}
    grads: dict[str, np.ndarray] = {}

    def __call__(self, x):
        self.inputs = x
        return self.forward(x)


    def update(self, learning_rate):
        for key in self.params:
            self.params[key] -= learning_rate * self.grads[key]


class Linear(Layer):

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        rng = np.random.default_rng()
        self.params = {
            'weight': rng.standard_normal((in_features, out_features)),
            'bias': rng.standard_normal((out_features)),
        }

    def forward(self, x):
        return np.dot(x, self.params['weight']) + self.params['bias']

    def backward(self, grad):
        self.grads = {
            'weight': np.dot(self.inputs.T, grad),
            'bias': np.sum(grad, axis=0),
        }
        
        return np.dot(grad, self.params['weight'].T)


class ReLU(Layer):

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, grad):
        return grad * (self.inputs > 0)


class Sigmoid(Layer):

    def forward(self, x):
        return 1/(1 + np.exp(-x))


    def backward(self, grad):
        sigmoid = 1/(1 + np.exp(-self.inputs))
        return grad * (1 - sigmoid) * sigmoid
    
    
    
class Model:

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

