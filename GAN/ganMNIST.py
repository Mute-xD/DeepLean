import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, LeakyReLU, Flatten, \
    BatchNormalization, Activation, Reshape, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import RMSprop
import struct


def setGPUMemoryGrowth():  # not working nothing changed
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print('No GPU Found')


setGPUMemoryGrowth()


class GanMNISTBase:
    def __init__(self):
        self.dis = None
        self.gen = None
        self.disMod = None
        self.advMod = None
        self.imgShape = (28, 28, 1)

    def discriminator(self):
        if self.dis is not None:
            return self.dis
        depth, dropout, = 64, 0.5
        self.dis = Sequential(name='dis')
        self.dis.add(Conv2D(depth * 1, 5, strides=2, padding='same', input_shape=self.imgShape))
        self.dis.add(LeakyReLU(alpha=0.2))
        self.dis.add(Dropout(dropout))

        self.dis.add(Conv2D(depth * 2, 5, strides=2, padding='same'))
        self.dis.add(LeakyReLU(alpha=0.2))
        self.dis.add(Dropout(dropout))

        self.dis.add(Conv2D(depth * 4, 5, strides=2, padding='same'))
        self.dis.add(LeakyReLU(alpha=0.2))
        self.dis.add(Dropout(dropout))

        self.dis.add(Conv2D(depth * 8, 5, strides=1, padding='same'))
        self.dis.add(LeakyReLU(alpha=0.2))
        self.dis.add(Dropout(dropout))

        self.dis.add(Flatten())
        self.dis.add(Dense(1, activation='sigmoid'))
        self.dis.summary()
        return self.dis



    def generator(self):
        if self.gen is not None:
            return self.gen
        depth, dropout, dim = 64 * 4, 0.5, 7
        self.gen = Sequential(name='gen')
        self.gen.add(Dense(dim * dim * depth, input_dim=100))
        self.gen.add(BatchNormalization(momentum=0.9))
        self.gen.add(Activation('relu'))
        self.gen.add(Reshape((dim, dim, depth)))
        self.gen.add(Dropout(dropout))

        self.gen.add(UpSampling2D())
        self.gen.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        self.gen.add(BatchNormalization(momentum=0.9))
        self.gen.add(Activation('relu'))

        self.gen.add(UpSampling2D())
        self.gen.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        self.gen.add(BatchNormalization(momentum=0.9))
        self.gen.add(Activation('relu'))

        self.gen.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        self.gen.add(BatchNormalization(momentum=0.9))
        self.gen.add(Activation('relu'))

        self.gen.add(Conv2DTranspose(1, 5, padding='same'))
        self.gen.add(Activation('sigmoid'))
        self.gen.summary()
        return self.gen

    def discriminatorModel(self):
        if self.disMod is not None:
            return self.disMod
        opt = RMSprop(learning_rate=2e-4, decay=6e-8)
        self.disMod = Sequential(name='disMod')
        self.disMod.add(self.discriminator())
        self.disMod.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
        return self.disMod

    def adversarialModel(self):
        if self.advMod is not None:
            return self.advMod
        opt = RMSprop(learning_rate=1e-4, decay=3e-8)
        self.advMod = Sequential(name='advMod')
        self.advMod.add(self.generator())
        self.dis.trainable = False  # <-
        self.advMod.add(self.discriminator())
        self.advMod.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
        return self.advMod


class GanMNIST:
    def __init__(self):
        self.base = GanMNISTBase()
        self.trainLabel, self.trainImage = None, None
        self.loadDataset()
        self.xTrain = self.trainImage
        self.xTrain = self.xTrain / 255.0
        self.xTrain = self.xTrain.reshape(-1, 28, 28, 1).astype(np.float32)
        self.dis = self.base.discriminatorModel()
        self.adv = self.base.adversarialModel()
        self.gen = self.base.generator()

    def loadDataset(self):
        with open('../MNIST_data/train-labels.idx1-ubyte', 'rb') as data:
            magic, n = struct.unpack('>II', data.read(8))
            self.trainLabel = np.fromfile(data, dtype=np.uint8)
        with open('../MNIST_data/train-images.idx3-ubyte', 'rb') as data:
            magic, num, row, col = struct.unpack('>IIII', data.read(16))
            self.trainImage = np.fromfile(data, dtype=np.uint8).reshape(len(self.trainLabel), -1)

        print('Dataset Loading Succeed!')

    def runner(self, trainStep=2000, batchSize=256):
        counter = 0
        while True:
            counter += 1
            imgReal = self.xTrain[np.random.randint(0, self.xTrain.shape[0], size=batchSize), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batchSize, 100])
            imgFake = self.gen.predict(noise)
            xData = np.concatenate((imgReal, imgFake))
            yLabel = np.ones([batchSize * 2, 1])
            yLabel[batchSize:, :] = 0
            disLoss = self.dis.train_on_batch(xData, yLabel)

            yLabel = np.ones([batchSize, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batchSize, 100])
            advLoss = self.adv.train_on_batch(noise, yLabel)

            logMsg = "%d: [D loss: %f, acc: %f]" % (counter, disLoss[0], disLoss[1])
            logMsg = "%s  [A loss: %f, acc: %f]" % (logMsg, advLoss[0], advLoss[1])
            print(logMsg)


if __name__ == '__main__':
    gan = GanMNIST()
    gan.runner(trainStep=10000, batchSize=256)
