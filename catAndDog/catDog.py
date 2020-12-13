import keras as k
import os


class CatAndDog:
    def __init__(self):
        self.model = k.models.Sequential()
        self.opt = k.optimizers.Adam(learning_rate=1e-4)
        self.batchSize = 32
        self.trainGen = k.preprocessing.image.ImageDataGenerator(rescale=1 / 255,
                                                                 shear_range=0.2,
                                                                 zoom_range=0.2,
                                                                 horizontal_flip=True)
        self.testGen = k.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
        self.trainLen = sum([len(files) for root, dirs, files in os.walk('./dataSet/train')])
        self.testLen = sum([len(files) for root, dirs, files in os.walk('./dataSet/test')])

    def loadDataset(self):
        self.trainGen = self.trainGen.flow_from_directory('./dataSet/train', target_size=(200, 200), batch_size=32)
        self.testGen = self.testGen.flow_from_directory('./dataSet/test', target_size=(200, 200), batch_size=32)

    def build(self):
        self.model.add(k.layers.Conv2D(128, 3, input_shape=(200, 200, 3), activation='relu'))
        self.model.add(k.layers.Conv2D(128, 3, activation='relu'))
        self.model.add(k.layers.MaxPooling2D(2))

        self.model.add(k.layers.Conv2D(64, 3, activation='relu'))
        self.model.add(k.layers.Conv2D(64, 3, activation='relu'))
        self.model.add(k.layers.MaxPooling2D(2))

        self.model.add(k.layers.Conv2D(32, 3, activation='relu'))
        self.model.add(k.layers.Conv2D(32, 3, activation='relu'))
        self.model.add(k.layers.MaxPooling2D(2))

        self.model.add(k.layers.Flatten())
        self.model.add(k.layers.Dense(64, activation='relu'))
        self.model.add(k.layers.Dropout(0.25))
        self.model.add(k.layers.Dense(2, activation='softmax'))

        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()

        self.model.fit_generator(self.trainGen,
                                 steps_per_epoch=self.trainLen/self.batchSize,
                                 epochs=50,
                                 validation_data=self.testGen,
                                 validation_steps=self.testLen/self.batchSize)


if __name__ == '__main__':
    catAndDog = CatAndDog()
    catAndDog.loadDataset()
    catAndDog.build()
