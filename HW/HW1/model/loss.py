import numpy as np



class MSE:


    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.square(y_pred - y_true))

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_pred.shape[0]


# class BinaryCrossEntropy(Loss):
class BinaryCrossEntropy:


    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(y_true * np.log(y_pred) +
                        (1 - y_true) * np.log(1 - y_pred))

    def backward(self):
        return (-np.divide(self.y_true, self.y_pred) + np.divide(
            1 - self.y_true, 1 - self.y_pred)) / self.y_pred.shape[0]