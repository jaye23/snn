import os
import time
import argparse
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

#**************定义SNN模型****************#
#定义一个单层全连接SNN模型
class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),#输入图像展平
            layer.Linear(28 * 28, 10, bias=False),#一个全连接层
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            # 使用 LIF 神经元（neuron.LIFNode）处理脉冲数据。使用反正切函数（surrogate.ATan()）作为替代函数支持反向传播。
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)
#**************定义SNN模型****************#

def main():
    '''
    :return: None

    * :ref:`API in English <lif_fc_mnist.main-en>`

    .. _lif_fc_mnist.main-cn:

    使用全连接-LIF的网络结构，进行MNIST识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_mnist.main-cn>`

    .. _lif_fc_mnist.main-en:

    The network with FC-LIF structure for classifying MNIST.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    '''
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    #初始化一个 ArgumentParser 对象，用于从命令行解析训练脚本的运行参数。
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    #设置模拟时间步长 T，即 SNN 仿真运行的时间片数量。每个输入在 SNN 网络中将被运行T个时间步，最终以脉冲频率（发放频率）作为输出。默认值100，整数
    parser.add_argument('-device', default='cuda:0', help='device')
    #指定模型和数据运行的计算设备，cuda:0（使用第一个 GPU）
    #使用CPU则修改为cpu
    parser.add_argument('-b', default=64, type=int, help='batch size')
    # 设置每次训练迭代的数据批次大小（BatchSize）。
    # 较大的批次大小通常能提高计算效率，但需要更大的显存容量。
    # 默认值： 64。
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    # 设置训练的总轮数（Epochs）。每个轮次表示网络对整个训练集完成一次遍历。
    # 默认值： 100。
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 指定数据加载器的线程数量，用于并行加载数据以加速训练过程。
    # 默认值： 4。
    parser.add_argument('-data-dir', type=str, help='root dir of MNIST dataset')
    #指定存放MNIST数据集的根目录。如果数据集不存在，则会自动下载到该目录。
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    # 指定训练日志和模型检查点（Checkpoint）保存的目录。
    # 默认值：./ logs。
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam',
                        help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    # 设置优化器的学习率（learningrate）。
    # 决定模型参数在每次梯度更新时的步长。
    # 默认值： 1e-3（0.001）。
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')
    # 设置LIF神经元模型中的时间常数τ，控制膜电位的衰减速度。
    # 默认值： 2.0。

    args = parser.parse_args()
    print(args)
    # 解析命令行中传入的参数，并将结果存储到args对象中。
    # 打印解析后的参数，便于调试。


    net = SNN(tau=args.tau)
    #使用用户指定的参数tau初始化SNN模型。
    print(net)

    net.to(args.device)
    # 将模型加载到指定设备上以加速训练和推理

    # *********************初始化数据加载器，加载MNIS数据集******************
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    # *********************初始化数据加载器，加载MNIS数据集******************

    #优化器和混合精度设置
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1 #初始化最高测试准确率

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    #检查点加载与日志记录
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

#****************************************************训练循环的核心代码*********************************************#
    for epoch in range(start_epoch, args.epochs):
        #开始一个训练周期，epoch 表示当前的训练轮次。 start_epoch: （0）训练的起始轮次（可能支持断点续训）。args.epochs: （100）总训练轮次。
        start_time = time.time()
        #记录当前时间，用于计算一个 epoch 的耗时。
        net.train()
        #将网络设置为训练模式。
        train_loss = 0
        train_acc = 0
        train_samples = 0
        #初始化当前 epoch 的train_loss: 累计损失。train_acc: 累计分类准确数。train_samples: 累计样本数量。
        for img, label in train_data_loader:
            #遍历训练数据加载器train_data_loader，逐批次获取输入数据img和对应标签label
            optimizer.zero_grad()#梯度清零，清除上一次迭代中累积的梯度，以防止梯度叠加导致错误的优化。
            img = img.to(args.device)
            label = label.to(args.device)
            #将图像和标签数据加载到计算设备（如GPU或CPU）。
            label_onehot = F.one_hot(label, 10).float()
            #将分类标签（标量形式）转换为 One-Hot 编码（向量形式）。10：分类类别数

            if scaler is not None:
                with amp.autocast(): #如果电脑支持“混合精度训练”（用更小的数字表示方式），就用它来加速计算和节省内存。

                    out_fr = 0.
                    for t in range(args.T):
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    out_fr = out_fr / args.T

                    loss = F.mse_loss(out_fr, label_onehot)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T

                loss = F.mse_loss(out_fr, label_onehot)

                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
# *********************************训练循环的核心代码*********************************************#

        #训练完成后的统计和记录
        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
        # 计算训练速度：每秒处理多少张图片。
        # 计算平均损失：累积的损失除以训练样本总数。
        # 计算平均准确率：累积的正确预测数除以训练样本总数。

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        # 使用日志工具writer记录本轮训练的损失（train_loss）和准确率（train_acc）

#*************************** 验证循环***********************
        net.eval()
        #把网络切换到“验证模式”（net.eval()），在这个模式下，模型不会更新参数。
        test_loss = 0
        test_acc = 0
        test_samples = 0
        #初始化验证过程中的统计变量
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                #更新验证统计信息
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
# *************************** 验证循环***********************

        #模型保存
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            #如果当前验证准确率（test_acc）比之前的最高准确率（max_test_acc）更高，就更新max_test_acc。
            save_max = True
            #设置 save_max = True，表示这是目前表现最好的模型，需要保存。

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
            #如果当前模型是最佳表现，保存为 checkpoint_max.pth。

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
        #无论是否是最佳表现，都保存为 checkpoint_latest.pth，方便断点续训。

        print(args)
        print(out_dir)
        print(
            f'epoch ={epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 保存绘图用数据
    net.eval()
    # 注册钩子
    output_layer = net.layer[-1]  # 输出层，获取网络的最后一层（输出层），net.layer[-1] 表示最后一层。
    output_layer.v_seq = []
    output_layer.s_seq = []
    #给输出层添加 v_seq 和 s_seq 属性，用来存储神经元的膜电位序列和脉冲序列。

    def save_hook(m, x, y): #钩子是一种特殊的函数，在网络运行时可以“偷偷观察”某些层的数据。
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))
        #这里的钩子会在每次前向传播时，记录输出层的膜电位（v）和脉冲（y）。

    output_layer.register_forward_hook(save_hook)
    #把钩子绑定到输出层，确保每次网络运行时自动调用 save_hook。

    with torch.no_grad():#表示接下来不需要计算梯度（因为我们只做测试）。
        img, label = test_dataset[0]#从测试数据集中随机取一张图片（img）和它的标签（label）。
        img = img.to(args.device)
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')#输出每个类别的脉冲频率，用来分析网络对图片的分类信心。

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        #把之前通过钩子函数记录的 v_seq（膜电位序列）和 s_seq（脉冲序列）合并成一个完整的数据结构。
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy", v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy", s_t_array)


if __name__ == '__main__':
    main()