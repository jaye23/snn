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

        self.device = device  # ADDED: Store device

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
        conv.append(layer.Linear(channels * 4 * 4, 512))
        # *********************从这里加channel*******************
        self.conv_layers = nn.Sequential(*conv)  # CNN 处理 DVS 数据

        # **多径信道**
        self.channel = MultiPathChannel(PDP=PDP, snr=snr, device=device)

        # self.conv_fc = nn.Sequential(
        #       # 第一个全连接层，把卷积层输出的特征映射到 512 维。
        #     layer.Linear(512, 512),  # 第一个全连接层，把卷积层输出的特征映射到 512 维。
        #     spiking_neuron(*args, **kwargs),  # 加入脉冲神经元，使得 FC 层也能处理时间信息。
        #
        #     layer.Dropout(0.5),
        #     layer.Linear(512, 110),  # 第二个全连接层，把 512 维的数据映射到 110 维（110 个类别）。
        #     spiking_neuron(*args, **kwargs),
        #
        #     layer.VotingLayer(10)  # 最终进行投票分类，把 110 维的结果转换成 10 维，适配 DVS 手势数据集。
        # )

        self.fc1 = layer.Linear(512, 512, bias=False)  # 假设与之前的风格一致，bias=False
        self.sn1 = spiking_neuron(*args, **kwargs)  # spiking_neuron_callable 是您传入的神经元类型
        self.dropout = layer.Dropout(0.5)
        self.fc2 = layer.Linear(512, 110, bias=False)
        self.sn2 = spiking_neuron(*args, **kwargs)  # <--- 这是新的目标脉冲神经元层
        self.voting = layer.VotingLayer(10)

        # --- ADDED: FSR 统计属性 for the target FC spiking layer ---
        self.spike_counts = torch.zeros(110, device=self.device)
        self.time_steps = 0  # 用于累积 T * B
        self.register_buffer('pruning_mask_fc', torch.ones(self.spike_counts))#新增：注册一个pruning_mask。register_buffer会将其作为模型状态的一部分

    def forward(self, x: torch.Tensor):
        #     return self.conv_fc(x)
        # # 输入x：事件流数据（batch_size, 2, H, W）。
        # # 通过conv_fc进行计算。
        # # 返回最终分类结果（10类）。
        #print("Shape before conv_layers:", x.shape)
        x = self.conv_layers(x)  # CNN 处理 DVS 事件
        #print("Shape after conv_layers:", x.shape)
        batch_size = x.shape[1]
        T = x.shape[0]
        #print("batch_size:", batch_size)
        #print("Shape after permute:", x.shape)
        x = x.view(batch_size*T, -1)  # 变成 1D 以适应信道 `[Batch, Feature_Size]`
        #print("Shape after view:", x.shape)
        x = self.channel(x)  # 通过多径信道
        #print("Shape after channel:", x.shape)  # 打印 `x` 的实际形状
        x = x.view(T, batch_size, 512)  # 变回 `[T, Batch, Channels, H, W]`
        #print("Shape after view:", x.shape)
        x = self.fc1(x)
        x = self.sn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        spikes_outputs = self.sn2(x)  # [T, B, 110] [16, 16, 110]
        # print("Shape after sn2:",spikes_outputs.shape)

        # 新增：应用剪枝掩码
        # self.pruning_mask_fc 形状是 [110]，会自动广播到 [T, B, 110]
        pruned_spikes = spikes_outputs * self.pruning_mask_fc

        #ADD：加入正则项
        # .sum() 将所有维度的脉冲加起来，得到一个标量值
        total_spikes_in_batch = pruned_spikes.sum()
        total_tbn = pruned_spikes.numel()
        total_spikes_in_batch = total_spikes_in_batch/total_tbn

        # --- ADDED: FSR 统计逻辑 for the target FC spiking layer ---
        if pruned_spikes is not None:
            # spikes_outputs 形状已经是 [T, B, num_neurons]
            # 对T和B维度求和，得到每个神经元的总脉冲数 (累加到模型属性)
            self.spike_counts += pruned_spikes.sum(dim=[0, 1]).detach()
            # 累积 T * B
            self.time_steps += pruned_spikes.shape[0] * pruned_spikes.shape[1]

        # fsr_per_neuron_epoch = self.spike_counts / self.time_steps
        # print('FSR:',fsr_per_neuron_epoch)

        x = self.voting(pruned_spikes)
        return x,total_spikes_in_batch
        # x = self.conv_fc(x)  # 进入全连接层进行分类
        # return x

    def reactivate_fading_neurons(self, fsr_threshold=0.07):
        """
        计算目标全连接层后脉冲神经元的FSR，并为低于阈值的神经元重新初始化输入权重。
        这是一个类方法，直接操作 self。

        Args:
            total_observation_time_steps_per_neuron (int): 在一个epoch中，单个神经元被观察的总时间步数。
            fsr_threshold (float): FSR的阈值，低于此值的神经元被视为不活跃。
        """
        print("\n--- [再激活检查开始] ---")

        # 2. 计算FSR
        if self.time_steps > 0:
            # 计算每个神经元的FSR
            fsr_per_neuron = self.spike_counts / self.time_steps
            print('FSR before reactivate:', fsr_per_neuron)

            # 3. 识别不活跃（衰减）的神经元
            fading_neuron_indices = torch.where(fsr_per_neuron < fsr_threshold)[0] #eg.tensor([1, 3]),索引为 1 和 3 的位置是不活跃神经元

            if len(fading_neuron_indices) > 0:
                print(
                    f"    检测到 {len(fading_neuron_indices)} 个不活跃神经元 (FSR < {fsr_threshold})。正在重新初始化输入权重...")

                # 4. 获取要修改的权重层 (现在通过 self 访问)
                linear_layer_to_reactivate = self.fc2

                # 在不计算梯度的上下文中执行权重修改
                with torch.no_grad():
                    for neuron_idx in fading_neuron_indices:
                        # 重新初始化连接到这个不活跃神经元的输入权重
                        # linear_layer_to_reactivate.weight 的形状是 [110, 512]
                        # 我们要重新初始化第 neuron_idx 行
                        torch.nn.init.kaiming_uniform_(linear_layer_to_reactivate.weight[neuron_idx, :].unsqueeze(0),
                                                       a=math.sqrt(5))

                        # # 如果该层有偏置项，也可以重新初始化
                        # if linear_layer_to_reactivate.bias is not None:
                        #     linear_layer_to_reactivate.bias[neuron_idx].data.fill_(0.0)

                print(f"    {len(fading_neuron_indices)} 个神经元的输入权重已重新初始化。")
            else:
                print(f"    所有目标神经元的FSR均高于阈值 {fsr_threshold}，无需再激活。")
        else:
            print("    观察总时间步数为零，无法计算FSR。")

        print("--- [再激活检查结束] ---\n")

    def prune_fading_neurons(self, fsr_threshold=0.07):
        fsr_per_neuron = self.spike_counts / self.time_steps
        print('FSR before prune:', fsr_per_neuron)

        neurons_to_prune_indices = torch.where(fsr_per_neuron < fsr_threshold)[0]

        if len(neurons_to_prune_indices) > 0:
            print(f"    根据最终FSR评估，将剪掉 {len(neurons_to_prune_indices)} 个不活跃神经元。")
            # 创建一个新的掩码，默认所有神经元都保留 (值为1)
            new_mask = torch.ones_like(self.pruning_mask_fc)
            # 将需要剪掉的神经元对应位置设为0
            new_mask[neurons_to_prune_indices] = 0.0

            # 更新模型中的掩码
            self.pruning_mask_fc.data = new_mask

            # 统计剪枝后的保留神经元比例
            retained_ratio = self.pruning_mask_fc.sum() / self.num_neurons_target_fc_layer
            print(f"    剪枝完成。神经元保留率: {retained_ratio.item():.2%}")
        else:
            print("    没有神经元的FSR低于阈值，无需剪枝。")

        print("--- [剪枝操作结束] ---\n")