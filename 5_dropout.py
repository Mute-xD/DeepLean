import keras as k


class MNIST:
    def __init__(self):
        self.model = k.models.Sequential()
        self.opt = k.optimizers.SGD(learning_rate=0.2)
        (self.xTrain, self.yTrain), (self.xTest, self.yTest) = k.datasets.mnist.load_data()

    def preProcess(self):
        self.xTrain = self.xTrain.reshape(self.xTrain.shape[0], -1) / 255.0
        self.xTest = self.xTest.reshape(self.xTest.shape[0], -1) / 255.0

        self.yTrain = k.utils.to_categorical(self.yTrain)
        self.yTest = k.utils.to_categorical(self.yTest)

    def build(self):
        self.model.add(k.layers.Dense(200, input_dim=784, activation='tanh'))
        self.model.add(k.layers.Dense(100, 'tanh'))
        self.model.add(k.layers.Dropout(0.25))
        self.model.add(k.layers.Dense(10, 'softmax'))
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.xTrain, self.yTrain, batch_size=16, epochs=20)
        loss, acc = self.model.evaluate(self.xTest, self.yTest)
        print('\nTest Set:\n', 'Loss: ', loss, '  Acc: ', acc)


if __name__ == '__main__':
    mnist = MNIST()
    print('\nxTrain: ', mnist.xTrain.shape, '\nyTrain: ', mnist.yTrain.shape,
          '\nxTest: ', mnist.xTest.shape, '\nyTest: ', mnist.yTest.shape)
    mnist.preProcess()
    mnist.build()
