import keras as k


class ModelSave:
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
        self.model.add(k.layers.Dense(units=10, input_dim=784, activation='softmax'))
        self.model.compile(optimizer=self.opt, loss='mse', metrics=['accuracy'])
        self.model.fit(self.xTrain, self.yTrain, batch_size=16, epochs=20)
        loss, acc = self.model.evaluate(self.xTest, self.yTest)
        print('\nTest Set:\n', 'Loss: ', loss, '  Acc: ', acc)
        self.model.save('savedModel.h5')


class ModelLoad:
    def __init__(self):
        self.model = None
        (self.xTrain, self.yTrain), (self.xTest, self.yTest) = k.datasets.mnist.load_data()

    def preProcess(self):
        self.xTrain = self.xTrain.reshape(self.xTrain.shape[0], -1) / 255.0
        self.xTest = self.xTest.reshape(self.xTest.shape[0], -1) / 255.0

        self.yTrain = k.utils.to_categorical(self.yTrain)
        self.yTest = k.utils.to_categorical(self.yTest)

    def loader(self):
        self.model = k.models.load_model('savedModel.h5')
        loss, acc = self.model.evaluate(self.xTest, self.yTest)
        print('\nTest Set:\n', 'Loss: ', loss, '  Acc: ', acc)


if __name__ == '__main__':
    # mnist = ModelSave()
    # print('\nxTrain: ', mnist.xTrain.shape, '\nyTrain: ', mnist.yTrain.shape,
    #       '\nxTest: ', mnist.xTest.shape, '\nyTest: ', mnist.yTest.shape)
    # mnist.preProcess()
    # mnist.build()
    load = ModelLoad()
    load.preProcess()
    load.loader()
