import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import tensorboardX


class NonLinearRegression:
    def __init__(self):
        self.writer = self.setWriter()
        self.xData = None
        self.yData = None

    def build(self):
        self.xData, self.yData = self.genDataset()
        xData, yData = torch.tensor(self.xData, dtype=torch.float), torch.tensor(self.yData, dtype=torch.float)
        model = NonLinearRegressionModel()
        mse_loss = nn.MSELoss()
        opt = optim.SGD(model.parameters(), lr=0.05)
        for i in range(1, 1001):
            out = model(xData)
            loss = mse_loss(out, yData)
            self.writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=i)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 100 == 0:
                print(i, loss.item())
        xPredict = torch.tensor(np.linspace(-5, 5, 200).reshape(-1, 1), dtype=torch.float)
        yPredict = model(xPredict)
        plt.scatter(self.xData, self.yData)
        plt.plot(xPredict.data.numpy(), yPredict.data.numpy(), 'r-')
        plt.show()

    @staticmethod
    def genDataset():
        xData = np.random.randn(100) - 0.5
        noise = np.random.normal(-0.2, 0.2, size=xData.size)
        yData = xData ** 2 + noise
        # plt.scatter(xData, yData)
        # plt.show()
        xData.shape = (-1, 1)  # 原地
        yData.shape = (-1, 1)
        return xData, yData

    @staticmethod
    def setWriter():
        return tensorboardX.SummaryWriter(comment='_nonLinear')


class NonLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 -> 10 -> 1
        self.layer1 = nn.Linear(1, 10)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh(x)
        out = self.layer2(x)
        return out


if __name__ == '__main__':
    nonLinear = NonLinearRegression()
    nonLinear.build()
