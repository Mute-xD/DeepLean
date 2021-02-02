import torch

"""
BP 反向传播
"""
x = torch.ones((2, 2), requires_grad=True)
print(x)
y = x + 2
z = (y ** 2) * 3
out = z.mean()
print(out)
out.backward()  # 反传
print(x.grad)  # 求x偏导
