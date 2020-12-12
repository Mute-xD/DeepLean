import keras as k
'''

pydot graphviz needed
pip install pydot

'''


class Plot:
    def __init__(self):
        self.model = k.models.Sequential()
        self.opt = k.optimizers.Adam(learning_rate=.001)
        (self.xTrain, self.yTrain), (self.xTest, self.yTest) = k.datasets.mnist.load_data()

    def preProcess(self):
        self.xTrain = self.xTrain.reshape(self.xTrain.shape[0], 28, 28, 1) / 255.0
        self.xTest = self.xTest.reshape(self.xTest.shape[0], 28, 28, 1) / 255.0

        self.yTrain = k.utils.to_categorical(self.yTrain)
        self.yTest = k.utils.to_categorical(self.yTest)

    def build(self):
        self.model.add(k.layers.Conv2D(filters=32,
                                       kernel_size=5,
                                       input_shape=(28, 28, 1),
                                       activation='relu',
                                       strides=1,
                                       padding='same'))
        self.model.add(k.layers.MaxPool2D(pool_size=2, strides=2, padding='same'))
        self.model.add(k.layers.Conv2D(64, 5, 1, 'same', activation='relu'))
        self.model.add(k.layers.MaxPool2D(2, 2, 'same'))
        self.model.add(k.layers.Flatten())
        self.model.add(k.layers.Dense(1024, activation='relu'))
        self.model.add(k.layers.Dropout(0.5))
        self.model.add(k.layers.Dense(10, activation='softmax'))
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['acc'])
        self.model.fit(self.xTrain, self.yTrain, batch_size=64, epochs=10)
        loss, acc = self.model.evaluate(self.xTest, self.yTest)
        print('\nTest Set:\n', 'Loss: ', loss, '  Acc: ', acc)

    def plotter(self):
        k.utils.plot_model(self.model)


if __name__ == '__main__':
    plotter = Plot()
    plotter.preProcess()
    plotter.build()
    plotter.plotter()
