import numpy as np

class Linear:
    def __init__(self, n_input, n_output, use_bias=True):
        self.use_bias = use_bias
        self.weight = np.random.normal(0, 0.1, (n_input, n_output))
        self.weight_grad = np.zeros_like(self.weight)
        if self.use_bias:
            self.bias = np.random.normal(0, 0.1, n_output)
            self.bias_grad = np.zeros_like(self.bias)

    def forward(self, x):
        self.data_in = x
        data_out = x @ self.weight
        if self.use_bias:
            data_out += self.bias
        return data_out

    def backward(self, e):
        self.weight_grad += self.data_in.T @ e
        if self.use_bias:
            self.bias_grad += e.sum(axis=0)
        return e @ self.weight.T

    def get_params_and_grads(self):
        if self.use_bias:
            params = np.concatenate((self.weight.ravel(), self.bias.ravel()))
            grads = np.concatenate((self.weight_grad.ravel(), self.bias_grad.ravel()))
            return params, grads
        else:
            return self.weight.ravel(), self.weight_grad.ravel()

    def set_params(self, params):
        if self.use_bias:
            weight_size = self.weight.shape[0] * self.weight.shape[1]
            self.weight = params[:weight_size].reshape(self.weight.shape)
            self.bias = params[weight_size:].reshape(self.bias.shape)
            self.weight_grad = np.zeros_like(self.weight)
            self.bias_grad = np.zeros_like(self.bias)
        else:
            self.weight = params.reshape(self.weight.shape)
            self.weight_grad = np.zeros_like(self.weight)

class ReLU:
    def forward(self, x):
        self.grad = x >= 0
        self.grad[~self.grad] = 0
        return x * self.grad
    def backward(self, e):
        return e * self.grad
    def get_params_and_grads(self):
        return np.array([]), np.array([])
    def set_params(self, params):
        pass

class MNISTModel:
    def __init__(self, hidden_size, n_class):
        self.modules = [
            Linear(28*28, hidden_size),
            ReLU(),
            Linear(hidden_size, n_class)
        ]
        self.batch_size = 0
        self.param_size = [
            module.get_params_and_grads()[0].shape[0] for module in self.modules
        ]

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, e):
        self.batch_size += e.shape[0]
        for module in reversed(self.modules):
            e = module.backward(e)
        return e

    def get_params_and_grads(self):
        params, grads = [], []
        for module in self.modules:
            param, grad = module.get_params_and_grads()
            params.append(param)
            grads.append(grad)
        return np.concatenate(params), np.concatenate(grads) / (self.batch_size + 1e-8)

    def set_params(self, params):
        offset = 0
        self.batch_size = 0
        for index, module in enumerate(self.modules):
            length = self.param_size[index]
            module.set_params(params[offset : offset + length])
            offset += length
