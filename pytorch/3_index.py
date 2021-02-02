import torch
tensor = torch.arange(2, 12, step=0.5)
print(tensor)
print(tensor[1:4])
index = [1, 3, 4, 5, 5]
print(tensor[index])
for i in tensor:
    print(i)
