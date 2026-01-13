import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math


# -------------------------------
# 定义全连接网络 FCN
# 输入 1 维 x，输出 2 维 [u,q]
# -------------------------------
class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = torch.tanh

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)  # 输出 2 维: u,q


# -------------------------------
# 实例化网络
# -------------------------------
net = FCN([1, 50, 50, 50, 2])


# -------------------------------
# 训练数据生成
# -------------------------------
def gen_traindata(num):
    xvals = np.linspace(-1, 1, num).reshape(num, 1)
    uvals = np.sin(np.pi * xvals)
    return xvals, uvals


# PDE 内部点
N_f = 200
x_f = torch.linspace(-1, 1, N_f).view(-1, 1)
# 观测数据点
N_m = 20
x_m, u_m = gen_traindata(N_m)
x_m = torch.tensor(x_m, dtype=torch.float32)
u_m = torch.tensor(u_m, dtype=torch.float32)

# PDE 点与观测点合并
x_pde = torch.cat([x_f, x_m], dim=0)

# 边界点
x_bc = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)


# -------------------------------
# PDE 残差函数
# -------------------------------
def pde_residual(x):
    x.requires_grad_(True)
    y = net(x)
    u = y[:, 0:1]
    q = y[:, 1:2]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]

    # 使用 -u_xx + q，与解析解符号一致
    return -u_xx + q


# -------------------------------
# 优化器和训练循环
# -------------------------------
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
num_epochs = 20000

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # PDE 残差损失
    res = pde_residual(x_pde)
    loss_pde = torch.mean(res ** 2)

    # 边界条件损失
    u_bc = net(x_bc)[:, 0:1]
    loss_bc = torch.mean(u_bc ** 2)

    # 数据损失
    u_pred_m = net(x_m)[:, 0:1]
    loss_data = torch.mean((u_pred_m - u_m) ** 2)

    # 总损失，使用自定义权重系数
    loss = 1.0 * loss_pde + 100.0 * loss_bc + 1000.0 * loss_data
    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        print(
            f"Epoch {epoch}, total loss {loss.item():.3e}, "
            f"PDE {loss_pde.item():.3e},"
            f" BC {loss_bc.item():.3e}, "
            f"observation data {loss_data.item():.3e}")

# -------------------------------
# 预测和可视化
# -------------------------------
x_test = torch.linspace(-1, 1, 500).view(-1, 1)
y_pred = net(x_test).detach().numpy()
u_pred = y_pred[:, 0:1]
q_pred = y_pred[:, 1:2]

x_test_np = x_test.numpy()
u_true = np.sin(np.pi * x_test_np)
q_true = -np.pi ** 2 * np.sin(np.pi * x_test_np)

# L2 相对误差
l2_u = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
l2_q = np.linalg.norm(q_true - q_pred) / np.linalg.norm(q_true)
print(f"L2 relative error u: {l2_u:.3e}, q: {l2_q:.3e}")

# 绘图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_test_np, u_true, "-", label="u_true")
plt.plot(x_test_np, u_pred, "--", label="u_NN")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.title("u(x)")

plt.subplot(1, 2, 2)
plt.plot(x_test_np, q_true, "-", label="q_true")
plt.plot(x_test_np, q_pred, "--", label="q_NN")
plt.xlabel("x")
plt.ylabel("q(x)")
plt.legend()
plt.title("q(x)")
plt.tight_layout()
plt.show()
