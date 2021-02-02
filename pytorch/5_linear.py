import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import tensorboardX


class LinearRegression:
    def __init__(self):
        self.xData, self.yData = None, None
        self.writer = None
        self.setSummary()

    def genData(self):
        xData = np.random.rand(100)
        noise = np.random.normal(0, 0.01, size=xData.shape)
        yData = 0.1 * xData + 0.2 + noise
        self.xData, self.yData = xData, yData

    def preProcess(self):
        xData = self.xData.reshape(-1, 1)
        yData = self.yData.reshape(-1, 1)
        return torch.tensor(xData, dtype=torch.float), torch.tensor(yData, dtype=torch.float)

    def build(self):
        xData, yData = self.preProcess()
        model = LinearRegressionModel()
        mse_loss = nn.MSELoss()
        opt = optim.SGD(model.parameters(), lr=0.1)
        for name, parameter in model.named_parameters():
            print(name, parameter)
        for i in range(1, 501):
            out = model(xData)
            loss = mse_loss(out, yData)
            self.writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=i)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 200 == 0:
                print(i, loss.item())
        for name, parameter in model.named_parameters():
            print(name, parameter)
        yPredict = model(xData)
        plt.scatter(self.xData, self.yData)
        plt.plot(self.xData, yPredict.data.numpy(), 'r-')
        plt.show()

    def setSummary(self):
        self.writer = tensorboardX.SummaryWriter()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)  # like dense

    def forward(self, x):  # 反传自动生成
        out = self.fc(x)
        return out


if __name__ == '__main__':
    linear = LinearRegression()
    linear.genData()
    linear.build()
