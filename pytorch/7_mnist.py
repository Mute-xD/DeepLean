import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as u_data


class MNIST:
    def __init__(self):
        self.writer = self.setWriter()
        self.batchSize = 64

    def build(self):
        trainPipe, testPipe = self.loadDataset()
        model = MNISTModel()
        mseLoss = nn.MSELoss()
        opt = optim.SGD(model.parameters(), lr=0.5)
        for epochs in range(1, 11):
            print('epochs', epochs)
            self.train(trainPipe, model, mseLoss, opt)
            self.test(testPipe, model)

    @staticmethod
    def train(trainPipe, model, mseLoss, opt):
        for i, data in enumerate(trainPipe):
            inputs, labels = data
            out = model(inputs)
            labels = labels.reshape(-1, 1)  # (64, 1)
            oneHot = torch.zeros(inputs.shape[0], 10).scatter(1, labels, 1)
            loss = mseLoss(out, oneHot)  # 两个参数shape一致
            opt.zero_grad()
            loss.backward()
            opt.step()

    @staticmethod
    def test(testPipe, model):
        correct = 0
        for i, data in enumerate(testPipe):
            inputs, labels = data
            out = model(inputs)
            _, predicted = torch.max(out, dim=1)  # maxVal, maxIndex
            for j in range(inputs.shape[0]):
                if predicted[j].item() == labels[j].item():
                    correct += 1
        acc = correct / 10000  # len(testset)
        print(acc)

    def loadDataset(self):
        trainSet = datasets.MNIST('./mnist', train=True, transform=transforms.ToTensor(), download=False)
        testSet = datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor(), download=False)

        trainPipe = u_data.DataLoader(dataset=trainSet, batch_size=self.batchSize, shuffle=True)
        testPipe = u_data.DataLoader(dataset=testSet, batch_size=self.batchSize, shuffle=True)
        return trainPipe, testPipe

    @staticmethod
    def setWriter():
        return tensorboardX.SummaryWriter(comment='_mnist')


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 10)  # 输入必须为二维
        self.softmax = nn.Softmax(dim=1)  # 目标维度 此处为(64, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)  # (64, 1, 28, 28) -> (64, 784)
        x = self.layer1(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    mnist = MNIST()
    mnist.build()
