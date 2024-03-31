import torch

A = torch.arange(20).reshape(5,4)
print(A)
print(A.T)

a = torch.tensor([1,2,3,4],dtype=torch.float32)
b = torch.tensor([1,2,3,4],dtype=torch.float32).reshape(4,1)
print('a:',a)
print('b:',b)
print(a*b)
print(torch.mul(a,b))
print(a@b)
print(torch.dot(a,a))
# print(torch.dot(b,b))
print(torch.mv(a*b,a))#矩阵与向量相乘
print(torch.mm(a*b,a*b))#必须为矩阵

#范数
print('a的范数:',torch.norm(a))
print('a*b的L1范数:',torch.abs(a*b).sum())#L1范数
print('a@b的L1范数:',torch.abs(a@b).sum())
print('a*b的f范数:',torch.norm(a*b))#佛罗贝尼乌斯范数Frobenius norm
print('a@b的f范数:',torch.norm(a@b))
print('')

z = torch.arange(24).reshape(2,3,4)
print(z)
print(z[:,-1:0,:])