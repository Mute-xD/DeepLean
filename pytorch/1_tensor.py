import torch
import numpy as np
a = torch.tensor([1, 2, 3], dtype=torch.int64)
print(a)
b = torch.tensor([4, 5, 6], dtype=torch.float64)
print(b)
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(tensor.ndim)  # 维度
print(tensor.shape)  # 形状

ones = torch.ones((2, 3))
print(ones)
zeros = torch.zeros((3, 3))
print(zeros)
rand = torch.rand((3, 4))
print(rand)
randint = torch.randint(0, 100, (2, 3))
print(randint)
randn = torch.randn((3, 4))  # 正态分布
print(randn)
a = torch.rand_like(tensor, dtype=torch.float64)
print(a)
b = a.view(6)  # reshape
print(b)
print(b[3].item())  # 单个值转Python scalars
print(np.array(b[:3]))  # 梦幻联动
array = np.array([1, 2, 3])
tensor = torch.tensor(array)
print(tensor)
