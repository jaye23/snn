# spikingjelly.activation_based.model.parametric_lif_net

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
import torch.nn.functional as F
import math



class MultiPathChannel(nn.Module):
    def __init__(self, PDP, snr, device):
        super(MultiPathChannel, self).__init__()
        self.PDP = PDP
        self.L = len(PDP)
        self.snr = 10 ** (snr / 10)
        self.device = device

    def forward(self, x):
        # sample a random channel and conv with the input data
        batch_size = x.size(0) #N*T L
        h = self.sample_h(batch_size)

        x_power = torch.norm(x.detach(), dim=1) / math.sqrt(x.size(1))
        # conv with the data
        x = x.unsqueeze(0)
        h = h.unsqueeze(1)

        # pad the input with zeros of dim L-1
        pad = ((self.L - 1) // 2, (self.L - 1) // 2)# L是单数情况

        x = F.pad(x, pad)

        #print("x before conv1d:", x.shape)
        #print("h before conv1d:", h.shape)
        output = F.conv1d(x, h, groups=batch_size).squeeze(0)

        #print("output after conv1d:", output.shape)
        # add gaussian noise
        noise = 1 / math.sqrt(2) * (torch.randn(output.size()))
        noise = noise.to(device=self.device)
        noise = noise / math.sqrt(self.snr) * x_power.unsqueeze(1)

        output += noise

        return output

    # 生成随机信道冲激响应（h）
    def sample_h(self, batch_size):
        h = 1 / math.sqrt(2) * (torch.randn(batch_size, self.L))
        # h 的形状为 [N, L]，其中 N 是 batch size，L 是多径信道的路径数。
        h = h.to(device=self.device)
        h = h * self.PDP.unsqueeze(0)

        return h



class DVSGestureNet(nn.Module):
    def __init__(self, channels, spiking_neuron: callable = None, PDP=None, snr=None, device=None, *args, **kwargs):
        # channels=128：每层卷积的通道数，默认是 128
        # spiking_neuron: callable = None：允许传入不同的脉冲神经元类型（如 LIF / PLIF），增强模型的灵活性。
        # PDP=None, snr=None, device=None: 如果没有提供，默认None
        # *args, **kwargs：支持传递额外参数，比如脉冲神经元的时间常数 τ。

        super().__init__()
        self.device = device
        # --- ADDED: FSR 统计属性 for the target FC spiking layer ---
        self.spike_counts = torch.zeros(512, device=self.device)
        self.time_steps = 0  # 用于累积 T * B
        # self.register_buffer('pruning_mask_fc',
        #                      torch.ones(self.spike_counts.shape))  # 新增：注册一个pruning_mask。register_buffer会将其作为模型状态的一部分
        # self.mask = torch.ones_like(self.pruning_mask_fc)
        self.mask = torch.ones(self.spike_counts.shape,device=self.device)
        conv = []
        for i in range(5):  # 5 层网络
            if conv.__len__() == 0:
                in_channels = 2  # 第一层输入通道数是 2（DVS 事件数据）
            else:
                in_channels = channels  # 其他层输入通道数是 channels (默认 128)

            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            # (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
            # kernel_size = 3, padding = 1 使得卷积后特征图大小不变。
            # bias = False 表示不使用偏置，因为后面有批归一化。
            conv.append(layer.BatchNorm2d(channels))  # 批归一化（BatchNorm2d）归一化数据，使得网络更稳定，避免梯度消失。
            conv.append(spiking_neuron(*args, **kwargs))  # 这里加入脉冲神经元
            conv.append(layer.MaxPool2d(2, 2))  # 每层后面加一个 2x2 的最大池化，减少数据维度，加快计算，同时保留关键特征。
        conv.append(layer.Flatten())  # 把 CNN 提取的特征展平，变成一个长向量，输入全连接层。
        conv.append(layer.Dropout(0.5))  # 在训练时随机丢弃 50% 的神经元，防止过拟合。
        # conv.append(layer.Linear(channels * 4 * 4, 512)) #([16, 16, 512])
        # conv.append(spiking_neuron(*args, **kwargs))#([16, 16, 512])
        # *********************从这里加channel*******************
        self.conv_layers = nn.Sequential(*conv)  # CNN 处理 DVS 数据

        self.fc = layer.Linear(channels * 4 * 4, 512)#([16, 16, 512])
        self.sn = spiking_neuron(*args, **kwargs)  #([16, 16, 512])

        self.channel = MultiPathChannel(PDP=PDP, snr=snr, device=device)
        # **多径信道**

        self.conv_fc = nn.Sequential(
              # 第一个全连接层，把卷积层输出的特征映射到 512 维。
            layer.Linear(512, 512),  # 第一个全连接层，把卷积层输出的特征映射到 512 维。
            spiking_neuron(*args, **kwargs),  # 加入脉冲神经元，使得 FC 层也能处理时间信息。

            layer.Dropout(0.5),
            layer.Linear(512, 110),  # 第二个全连接层，把 512 维的数据映射到 110 维（110 个类别）。
            spiking_neuron(*args, **kwargs),

            layer.VotingLayer(10)  # 最终进行投票分类，把 110 维的结果转换成 10 维，适配 DVS 手势数据集。
        )

    def forward(self, x: torch.Tensor):
        #     return self.conv_fc(x)
        # # 输入x：事件流数据（batch_size, 2, H, W）。
        # # 通过conv_fc进行计算。
        # # 返回最终分类结果（10类）。
        #print("Shape before conv_layers:", x.shape)
        x = self.conv_layers(x)  # CNN 处理 DVS 事件
        x = self.fc(x)
        x = self.sn(x)

        # add:神经元输出
        neuron_output = x

        # --- ADDED: FSR 统计逻辑 for the target FC spiking layer ---
        if x is not None:
            # spikes_outputs 形状已经是 [T, B, num_neurons]
            # 对T和B维度求和，得到每个神经元的总脉冲数 (累加到模型属性)
            self.spike_counts += x.sum(dim=[0, 1]).detach()
            # 累积 T * B
            self.time_steps += x.shape[0] * x.shape[1]

        # fsr_per_neuron_epoch = self.spike_counts / self.time_steps
        # print('FSR:',fsr_per_neuron_epoch)

        # 新增：应用剪枝掩码
        x = x * self.mask.view(1, 1, -1)

        # ADD：加入正则项
        # .sum() 将所有维度的脉冲加起来，得到一个标量值
        total_spikes_in_batch = x.sum()
        total_tbn = x.numel()
        total_spikes_in_batch = total_spikes_in_batch / total_tbn

        batch_size = x.shape[1]
        T = x.shape[0]
        #print("batch_size:", batch_size)
        #print("Shape after permute:", x.shape)
        x = x.view(batch_size*T, -1)  # 变成 1D 以适应信道 `[Batch, Feature_Size]`
        #print("Shape after view:", x.shape)
        x = self.channel(x)  # 通过多径信道
        #print("Shape after channel:", x.shape)  # 打印 `x` 的实际形状
        x = x.view(T, batch_size, 512)  # 变回 `[T, Batch, Channels, H, W]`
        # print("Shape after view:", x.shape)
        x = self.conv_fc(x)  # 进入全连接层进行分类
        return x,total_spikes_in_batch,neuron_output

    def fsr(self):
        fsr_per_neuron = (self.spike_counts / self.time_steps)* self.mask
        return fsr_per_neuron


    def prune_fading_neurons(self, fsr_threshold):
        fsr_per_neuron = (self.spike_counts / self.time_steps)* self.mask
        # print('FSR before prune:', fsr_per_neuron)

        neurons_to_prune_indices = torch.where(fsr_per_neuron < fsr_threshold)[0]

        if len(neurons_to_prune_indices) > 0:
            print(f"    根据最终FSR评估，将剪掉 {len(neurons_to_prune_indices)} 个不活跃神经元。")
            # 创建一个新的掩码，默认所有神经元都保留 (值为1)
            # 将需要剪掉的神经元对应位置设为0
            self.mask[neurons_to_prune_indices] = 0.0

            # print('mask:',self.mask)

            print('FSR after prune:', fsr_per_neuron * self.mask)

            # 统计剪枝后的保留神经元数量和总数量
            num_retained = int(self.mask.sum().item())
            num_total = self.mask.numel()

            # # 更新模型中的掩码
            # self.pruning_mask_fc = self.mask
            #
            # # 统计剪枝后的保留神经元数量和总数量
            # num_retained = int(self.pruning_mask_fc.sum().item())
            # num_total = self.pruning_mask_fc.numel()
            # 统计剪枝后的保留神经元比例
            retained_ratio = num_retained / num_total
            print(f"    剪枝完成。神经元保留率: {retained_ratio:.2%}")
            print(f"    剪枝完成。保留神经元数目: {num_retained}")

        else:
            print("    没有神经元的FSR低于阈值，无需剪枝。")

        print("--- [剪枝操作结束] ---\n")

        return num_retained,(num_total-num_retained),fsr_per_neuron,neurons_to_prune_indices

    def prune_fading_neurons_num(self, prune_num):
        fsr_per_neuron = (self.spike_counts / self.time_steps)* self.mask
        print('FSR before prune:', fsr_per_neuron)

        # 新增参数：prune_num 表示想剪掉的神经元数量
        prune_num = min(prune_num, fsr_per_neuron.numel())  # 防止超出总神经元数

        # 获取 fsr_per_neuron 中最小的 prune_num 个索引
        _, neurons_to_prune_indices = torch.topk(fsr_per_neuron, k=prune_num, largest=False)

        if len(neurons_to_prune_indices) > 0:
            print(f"    根据最终FSR评估，将剪掉 {len(neurons_to_prune_indices)} 个不活跃神经元。")
            # 创建一个新的掩码，默认所有神经元都保留 (值为1)
            # 将需要剪掉的神经元对应位置设为0
            self.mask[neurons_to_prune_indices] = 0.0

            print('mask:',self.mask)

            # 统计剪枝后的保留神经元数量和总数量
            num_retained = int(self.mask.sum().item())
            num_total = self.mask.numel()

            # # 更新模型中的掩码
            # self.pruning_mask_fc = self.mask
            #
            # # 统计剪枝后的保留神经元数量和总数量
            # num_retained = int(self.pruning_mask_fc.sum().item())
            # num_total = self.pruning_mask_fc.numel()
            # 统计剪枝后的保留神经元比例
            retained_ratio = num_retained / num_total
            print(f"    剪枝完成。神经元保留率: {retained_ratio:.2%}")
            print(f"    剪枝完成。保留神经元数目: {num_retained}")

        else:
            print("    没有神经元的FSR低于阈值，无需剪枝。")

        print("--- [剪枝操作结束] ---\n")

        return num_retained,(num_total-num_retained),fsr_per_neuron,neurons_to_prune_indices


