import numpy as np

class SoftmaxNLL:
    def loss(pred, label):
        max_pred = pred.max(axis=1, keepdims=True)
        return np.mean(
            np.log(np.exp(pred - max_pred).sum(axis=1) + max_pred.ravel() - pred[np.arange(pred.shape[0]), label])
        )
    def gradient(pred, label):
        exp_pred = np.exp(pred - pred.max(axis=1, keepdims=True))
        grad = exp_pred / exp_pred.sum(axis=1, keepdims=True)
        grad[np.arange(pred.shape[0]), label] -= 1
        return grad

class SGD:
    def __init__(self, model, lr, l2=0.0):
        self.model = model
        self.lr = lr
        self.l2 = l2
        self.multiplier = 1

    def update(self):
        params, grads = self.model.get_params_and_grads()
        grads += 2 * self.l2 * params
        self.model.set_params(params - self.lr * grads * self.multiplier)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

class ExpDecayLR:
    def __init__(self, init, alpha):
        self.init = init
        self.alpha = alpha

    def get_multiplier(self, cur_epoch):
        return self.init * self.alpha ** cur_epoch
