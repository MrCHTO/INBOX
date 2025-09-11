import torch
#创建一个未初始化的5x3矩阵
x=torch.empty(5,3)
print(x)

#创建一个随机初始化的5x3矩阵
x=torch.rand(5,3)
print(x)

#创建一个5x3的零矩阵，类型为long
x=torch.zeros(5,3,dtype=torch.long)
print(x)

#直接从数据创建tensor
x=torch.tensor([5.5,3])
print(x)

#创建一个tensor，并设置requires_grad=True以跟踪计算历史
x=torch.ones(2,2,requires_grad=True)
print(x)

#对tensor进行操作
y=x+2
print(y)

#y是操作的结果，所以它有grad_fn属性
print(y.grad_fn)

#对y进行更多操作
z=y*y*3
out=z.mean()

print(z,out)

a = torch.randn(1,2,3,4,5)
print(a)
