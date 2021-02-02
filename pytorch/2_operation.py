import torch
a = torch.randint(1, 5, (2, 3))
b = torch.randint(1, 5, (2, 3))
print(a, b)
print(a + b)
print(torch.add(a, b))

result = torch.zeros((2, 3))
torch.add(a, b, out=result)
print(result)
# 原地操作，加下划线
a.add_(b)
print(a)
a = a.float()  # 数据类型转换
print(a)
a = a.T  # 矩阵转置
print(a)
##############################################################################
# PART 2
sample = torch.rand((3, 2))
print(sample)
print(sample.sum())
print(sample.min())
print(sample.max())
print(sample.argmin())
print(sample.argmax())
print(sample.mean())
print(sample.median())  # 中位数
print(torch.sqrt(sample))  # 开方
print(sample ** 2)  # 平方
