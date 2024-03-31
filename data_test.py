import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
print(x.reshape(3, 4))

y = torch.zeros(12)
print(y)

z = torch.tensor([[1, 2, 3, 4],
                  [2, 3, 4, 5],
                  [3, 2, 3, 2]])
print(z)

x = torch.cat((x.reshape(3, 4), z), dim=0)
print(x)

print(x.sum())

# 广播 3*1 + 1*2 会广播成3*2+3*2,然后相加
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)

print(a, b)
print(a + b)

# 赋值
z = a + b

z[0:2, :] = 12
print(z)

# 指针id
X = torch.arange(12).reshape(3, 4)
Y = torch.arange(12, 24).reshape(3, 4)
print(X)
print(Y)

before = id(Y)
Y += X #id不变
print(id(Y) == before)

Z = torch.zeros_like(Y)
Z = X + Y
print('before:', before)
print('id(Z):', id(Z))

#numpy
A = X.numpy()
B = torch.tensor(A)
print(type(A),type(B))

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print((int(a)))