import keras as k


class RNN:
    def __init__(self):
        self.model = k.models.Sequential()
        self.opt = k.optimizers.Adam(learning_rate=.001)
        (self.xTrain, self.yTrain), (self.xTest, self.yTest) = k.datasets.mnist.load_data()
        print(self.xTrain.shape)

    def preProcess(self):
        self.xTrain = self.xTrain / 255.0
        self.xTest = self.xTest / 255.0

        self.yTrain = k.utils.to_categorical(self.yTrain)
        self.yTest = k.utils.to_categorical(self.yTest)

    def build(self):
        self.model.add(k.layers.recurrent.SimpleRNN(50, input_shape=(28, 28)))
        self.model.add(k.layers.Dense(10, activation='softmax'))
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['acc'])
        self.model.fit(self.xTrain, self.yTrain, batch_size=64, epochs=10)
        loss, acc = self.model.evaluate(self.xTest, self.yTest)
        print('\nTest Set:\n', 'Loss: ', loss, '  Acc: ', acc)


if __name__ == '__main__':
    rnn = RNN()
    rnn.preProcess()
    rnn.build()
