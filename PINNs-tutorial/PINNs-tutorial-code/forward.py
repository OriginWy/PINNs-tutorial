# -------------------------------
# 正问题 PINNs 完整示例
# -------------------------------

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# 1. 定义全连接神经网络
class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = torch.tanh

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# 2. 实例化网络
net_u = FCN([1, 20, 20, 20, 1])  # 输入1维，输出1维，隐藏层20个神经元

# 3. 定义 PDE 残差函数
def pde_residual(x):
    x.requires_grad_(True)
    u = net_u(x)

    # 计算一阶导数
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # 计算二阶导数
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # 已知源项函数 q(x) = -pi^2 sin(pi x)
    q = -math.pi**2 * torch.sin(math.pi * x)
    return u_xx - q

# 4. 训练数据：PDE 内部点 + 边界点
N_f = 100
x_f = torch.linspace(-1, 1, N_f).view(-1, 1)
x_bc = torch.tensor([[-1.0], [1.0]])

# 5. 定义优化器
optimizer = torch.optim.Adam(net_u.parameters(), lr=1e-3)

# 6. 训练循环
for epoch in range(5000):
    optimizer.zero_grad()

    # PDE 残差损失
    loss_pde = torch.mean(pde_residual(x_f)**2)
    # 边界条件损失
    loss_bc = torch.mean(net_u(x_bc)**2)

    # 总损失
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.3e}")

# 7. 结果可视化
x_test = torch.linspace(-1, 1, 200).view(-1, 1)
u_pred = net_u(x_test).detach().numpy()
u_true = np.sin(np.pi * x_test.numpy())

plt.plot(x_test, u_true, label="Exact")
plt.plot(x_test, u_pred, "--", label="PINN")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("PINNs Solution of the Forward Problem")
plt.show()
