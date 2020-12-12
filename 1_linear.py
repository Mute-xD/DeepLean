import keras as k
import numpy as np
import matplotlib.pyplot as plt


class Linear:
    def __init__(self):
        self.xData = np.random.rand(100)
        self.yOrigin = self.xData * 0.1 + 0.2
        noise = np.random.normal(0, 0.02, self.xData.shape)
        self.yData = self.yOrigin + noise
        self.model = k.models.Sequential()

    @staticmethod
    def plot(x_, y_):
        plt.scatter(x_, y_)
        plt.show()

    def build(self):
        self.model.add(k.layers.Dense(units=1, input_dim=1))
        self.model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
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
    linear = Linear()
    linear.build()
