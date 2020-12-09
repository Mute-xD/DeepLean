import keras as k
import numpy as np
import matplotlib.pyplot as plt


class NonLinear:
    def __init__(self):
        self.xData = np.linspace(-0.5, 0.5, 1000)
        self.yOrigin = np.square(self.xData)
        noise = np.random.normal(0, 0.02, self.xData.shape)
        self.yData = self.yOrigin + noise
        self.model = k.models.Sequential()
        self.opt = k.optimizers.SGD(lr=0.2)

    @staticmethod
    def plot(x_, y_):
        plt.scatter(x_, y_)
        plt.show()

    def build(self):
        self.model.add(k.layers.Dense(units=10, input_dim=1, activation='tanh'))
        self.model.add(k.layers.Dense(units=1, activation='tanh'))
        self.model.compile(optimizer=self.opt, loss='mse')
        # self.model.fit(x=self.xData, y=self.yData, epochs=2000)
        for step in range(3001):
            cost = self.model.train_on_batch(self.xData, self.yData)
            if step % 100 == 0:
                print('Step: ', step, '  Loss: ', cost)
        weight, bias = self.model.layers[0].get_weights()
        print('Weight: ', weight, '  Bias: ', bias)
        yPredict = self.model.predict(self.xData)
        plt.scatter(self.xData, self.yData)
        plt.scatter(self.xData, yPredict)
        plt.scatter(self.xData, self.yOrigin)
        plt.legend(['yData', 'yPredict', 'yOrigin'])
        plt.show()


if __name__ == '__main__':
    nonLinear = NonLinear()
    nonLinear.plot(nonLinear.xData, nonLinear.yData)
    nonLinear.build()
