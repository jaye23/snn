import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

class Neuron:
    def __init__(self):
        # 初始化膜电位为 0
        self.v = torch.tensor(0.0)

    def neuronal_charge(self, x: torch.Tensor):
        # 更新膜电位
        self.v = self.v + x


if_layer = neuron.IFNode(step_mode='s')

#print(if_layer.v)
x = torch.rand(size=[2, 3])  # 生成一个形状为 [2, 3] 的随机张量
if_layer(x)  # 将输入 x 传递给脉冲神经元层
#print(f'x.shape={x.shape}, if_layer.v.shape={if_layer.v.shape}')
# 输出：x.shape=torch.Size([2, 3]), if_layer.v.shape=torch.Size([2, 3])
if_layer.reset()  # 调用 reset() 方法重置神经元状态

x = torch.rand(size=[4, 5, 6])  # 生成一个形状为 [4, 5, 6] 的随机张量
if_layer(x)  # 将输入 x 传递给神经元层
#print(f'x.shape={x.shape}, if_layer.v.shape={if_layer.v.shape}')
# 输出：x.shape=torch.Size([4, 5, 6]), if_layer.v.shape=torch.Size([4, 5, 6])


# 创建一个神经元对象
neuron = Neuron()

# 查看初始膜电位
#print("初始膜电位：", neuron.v)  # 输出：初始膜电位： 0.0

# 输入信号
input_signal = torch.tensor(0.5)

# 更新膜电位
neuron.neuronal_charge(input_signal)

# 查看更新后的膜电位
#print("更新后的膜电位：", neuron.v)  # 输出：更新后的膜电位： 0.5y


if_layer.reset()
x = torch.as_tensor([0.02])
# 定义输入信号 x，其值为 0.02。这表示每次输入的充电量恒定为 0.02。
# torch.as_tensor 是 PyTorch 的一个方法，用来创建张量
T = 150
# 模拟的时间长度为 150 个时间步（T = 150）。
# 在每个时间步内，神经元都会接收输入信号 x 并更新膜电位。
s_list = []
v_list = []
# 创建两个空列表，用来存储每个时间步的脉冲信号（s_list）和膜电位（v_list）。
for t in range(T): #一个循环，从 t=0 到 t=149，
    s_list.append(if_layer(x)) #if_layer(x)的返回值是当前时间步的脉冲信号（spike，值为0或1）。
    v_list.append(if_layer.v)

#plot
dpi = 300
figsize = (12, 8)
visualizing.plot_one_neuron_v_s(torch.cat(v_list).numpy(), torch.cat(s_list).numpy(), v_threshold=if_layer.v_threshold,
                                v_reset=if_layer.v_reset,
                                figsize=figsize, dpi=dpi)
plt.show()


if_layer.reset()
T = 50
x = torch.rand([32]) / 8.
# 随机生成一个形状为 [32] 的输入张量，每个神经元的输入值为 [0,0.125)的随机值。
# 与单神经元的区别：单神经元时，输入张量的形状是 [1]，而这里是 [32]，表示每个时间步有 32 个神经元同时接收输入。
s_list = []
v_list = []
for t in range(T):
    s_list.append(if_layer(x).unsqueeze(0))
    v_list.append(if_layer.v.unsqueeze(0))

s_list = torch.cat(s_list)
v_list = torch.cat(v_list)
# s_list 的形状：[T, 32]（50 行，32 列）。
# v_list 的形状：[T, 32]（50 行，32 列）。
#单神经元时，拼接后的张量形状是 [T, 1]，只有一个神经元的数据。

figsize = (12, 8)
dpi = 200
visualizing.plot_2d_heatmap(array=v_list.numpy(), title='membrane potentials', xlabel='simulating step',
                            ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)


visualizing.plot_1d_spikes(spikes=s_list.numpy(), title='membrane sotentials', xlabel='simulating step',
                        ylabel='neuron index', figsize=figsize, dpi=dpi)

plt.show()


if_layer = neuron.IFNode()
print(f'if_layer.backend={if_layer.backend}') #查看计算使用后端
# if_layer.backend=torch

print(f'step_mode={if_layer.step_mode}, supported_backends={if_layer.supported_backends}')
# step_mode=s, supported_backends=('torch',)


if_layer.step_mode = 'm'
print(f'step_mode={if_layer.step_mode}, supported_backends={if_layer.supported_backends}')
# 多步模式下，IFNode 支持 cupy 后端（除了 PyTorch）。
# cupy 后端： 通过 CuPy（类似 NumPy，但运行在 GPU 上）实现更高效的计算，特别适合大型时间序列的计算任务。
# step_mode=m, supported_backends=('torch', 'cupy')

device = 'cuda:0'
if_layer.to(device)
if_layer.backend = 'cupy'  # switch to the cupy backend
print(f'if_layer.backend={if_layer.backend}')
# if_layer.backend=cupy

x_seq = torch.rand([8, 4], device=device) #[8, 4]：[T, N] 时间步长度 × 神经元数量。
#device=device：数据存储在 GPU 上（因为 device='cuda:0'）。
y_seq = if_layer(x_seq)
if_layer.reset()