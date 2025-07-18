import torch
import random
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import DVSNet_Channel_Pruning_Period_fisher

import torchvision.transforms as transforms
import time
import os
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np

class RandomTranslate:
    def __init__(self, max_offset=25):
        self.max_offset = max_offset

    def __call__(self, img):
        off1 = random.randint(-self.max_offset, self.max_offset)
        off2 = random.randint(-self.max_offset, self.max_offset)
        return transforms.functional.affine(img, angle=0.0, scale=1.0, shear=0.0, translate=(off1, off2))


class toTensor:
    def __call__(self, img):
        return torch.from_numpy(img).float()


def evaluate(net, test_data_loader, device, train_time):
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0

    with torch.no_grad():
        for frame, label in test_data_loader:
            frame = frame.to(device)
            frame = frame.transpose(0, 1)
            label = label.to(device)
            label_onehot = F.one_hot(label, 11).float()

            output_with_time, batch_spikes = net(frame)
            out_fr = output_with_time.mean(0)
            loss = F.mse_loss(out_fr, label_onehot)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

    test_time = time.time()
    test_speed = test_samples / (test_time - train_time)
    test_loss /= test_samples
    test_acc /= test_samples

    return test_loss, test_acc, test_speed

def compute_fisher_information(net, data_loader, device):
    net.eval()
    total_fisher = None
    sample_count = 0

    for frame, label in data_loader:
        frame = frame.to(device).transpose(0, 1)
        label = label.to(device)
        label_onehot = F.one_hot(label, 11).float()

        net.zero_grad()
        output_with_time, _, neuron_output = net(frame)
        neuron_output.retain_grad()

        out_fr = output_with_time.mean(0)
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()

        with torch.no_grad():
            batch_fisher = neuron_output.grad.pow(2).sum(dim=(0, 1))
            if total_fisher is None:
                total_fisher = batch_fisher
            else:
                total_fisher += batch_fisher
        sample_count += neuron_output.shape[0] * neuron_output.shape[1]

        functional.reset_net(net)

    return total_fisher / sample_count

def spearman_corr(x, y):
    # 将两个向量转换为排名
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))

    # 计算皮尔逊相关系数
    return np.corrcoef(x_rank, y_rank)[0, 1]


def main():
    # python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:1', help='device')
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-epochs', default=450, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default=os.path.join(current_dir, "datasets", "DVS"), type=str,
                        help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. sdg or adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-channels', default=64, type=int, help='channels of CSNN')
    parser.add_argument('-snr', default=10, type=int, help='snr')
    parser.add_argument('-pdp-values', type=float, nargs='+', default=[1.0, 0.5, 0.3, 0.2, 0.1],
                        help='custom PDP values for pruning schedule')
    parser.add_argument('-prune-acc-threshold', type=float, default=0.95,
                        help='accuracy threshold to trigger pruning')
    parser.add_argument('-prune-acc-stop', type=float, default=0.60,
                        help='accuracy threshold to stop pruning')
    parser.add_argument('-prune-loss-threshold', type=float, default=0.0175,
                        help='loss threshold to trigger pruning')
    parser.add_argument('-fsr-threshold', type=float, default=0.25,
                        help='FSR threshold for pruning')  # 改一下
    parser.add_argument('-regular-lambda', type=float, default=1e-3,
                        help='regularization strength')  # 改一下1e-3  shell脚本跑多个参数，保留神经元到100以下
    parser.add_argument('-min-neurons-threshold', type=int, default=10,
                        help='minimum number of neurons to stop pruning')
    parser.add_argument('-prune-period', type=int, default=20,
                        help='pruning cycle in epochs (perform pruning every N epochs once triggered)')

    args = parser.parse_args()
    args.opt = "adam"
    print(args)

    # pdp_values = torch.tensor([1.0, 0.5, 0.3, 0.2, 0.1])  # 自定义 PDP
    pdp_values = torch.tensor(args.pdp_values)
    PDP = pdp_values / pdp_values.sum()  # 归一化
    PDP = PDP.to(args.device)  # 确保 PDP 在正确的计算设备上

    # --- 新增：定义标志位和触发阈值 ---
    pruning_started = False  # 标记开始剪枝
    prune_stopped = False  # 一旦 True，不再剪枝
    # prune_acc_threshold = 0.80  # 触发剪枝的准确率阈值
    # prune_acc_stop = 0.60  # 停止剪枝的准确率阈值
    # prune_loss_threshold = 0.0175  # 触发剪枝损失阈值
    # fsr_threshold = 0.07  # FSR阈值
    prune_epoch = []  # 新增：用于记录prune发生的epoch，-1代表还未发生
    neuron_acc_history = []
    # regular_lambda = 1e-4  # 新增：正则化强度超参数
    # min_neurons_threshold = 10  # 当保留的神经元数量小于或等于这个值时，将不再进行剪枝。

    net = DVSNet_Channel_Pruning_Period_fisher.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode,
                                                      surrogate_function=surrogate.ATan(), detach_reset=True, PDP=PDP,
                                                      snr=args.snr, device=args.device)

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    # print(net)

    net.to(args.device)

    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T,
                              split_by='number',
                              transform=transforms.Compose([toTensor(), RandomTranslate(max_offset=25)]))
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T,
                             split_by='number')

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    # writer = SummaryWriter(out_dir, purge_step=start_epoch)
    # with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
    #     args_txt.write(str(args))
    #     args_txt.write('\n')
    #     args_txt.write(' '.join(sys.argv))

    test_acc_list = []
    train_acc_list = []
    epoch_list = []
    pruning_count = 0
    first_prune_epoch = None  # 记录首次剪枝的 epoch
    neuron_acc_history = []
    spearman_list = []
    pearson_list = []
    last_prune_acc = None  # 每轮剪枝前一个 epoch 的 test_acc
    # 在训练循环开始之前，初始化当前的FSR阈值
    current_fsr_threshold = args.fsr_threshold
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 11).float()

            if scaler is not None:
                with amp.autocast():
                    # out_fr,batch_spikes = net(frame).mean(0)
                    # 1. 先接收模型返回的元组，解包到两个变量中
                    output_with_time, batch_spikes, neuron_output = net(frame)

                    # 2. 然后对需要处理的那个张量（即第一个返回值）调用 .mean(0)
                    out_fr = output_with_time.mean(0)
                    main_loss = F.mse_loss(out_fr, label_onehot)
                    if prune_epoch:
                        loss = main_loss + args.regular_lambda * batch_spikes
                    else:
                        loss = main_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # out_fr,batch_spikes = net(frame).mean(0)
                # 1. 先接收模型返回的元组，解包到两个变量中
                output_with_time, batch_spikes,neuron_output = net(frame)

                # 2. 然后对需要处理的那个张量（即第一个返回值）调用 .mean(0)
                out_fr = output_with_time.mean(0)
                main_loss = F.mse_loss(out_fr, label_onehot)
                if prune_epoch:
                    loss = main_loss + args.regular_lambda * batch_spikes
                else:
                    loss = main_loss

                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        # test
        test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device, train_time)
        last_prune_acc = test_acc

        # net.eval()
        # test_loss = 0
        # test_acc = 0
        # test_samples = 0
        # with torch.no_grad():
        #     for frame, label in test_data_loader:
        #         frame = frame.to(args.device)
        #         frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        #         label = label.to(args.device)
        #         label_onehot = F.one_hot(label, 11).float()
        #         # out_fr = net(frame).mean(0)
        #         # 1. 先接收模型返回的元组，解包到两个变量中
        #         output_with_time, batch_spikes = net(frame)
        #         # 2. 然后对需要处理的那个张量（即第一个返回值）调用 .mean(0)
        #         out_fr = output_with_time.mean(0)
        #         loss = F.mse_loss(out_fr, label_onehot)
        #         test_samples += label.numel()
        #         test_loss += loss.item() * label.numel()
        #         test_acc += (out_fr.argmax(1) == label).float().sum().item()
        #         functional.reset_net(net)
        # test_time = time.time()
        # test_speed = test_samples / (test_time - train_time)
        # test_loss /= test_samples
        # test_acc /= test_samples
        # # writer.add_scalar('test_loss', test_loss, epoch)
        # # writer.add_scalar('test_acc', test_acc, epoch)

        # if not pruning_started and test_loss <= args.prune_loss_threshold and test_acc >= args.prune_acc_threshold:

        if not pruning_started and test_acc >= args.prune_acc_threshold and epoch >= 150:
            # add
            fisher_info = compute_fisher_information(net, train_data_loader, args.device)

            # 启动第一次剪枝
            num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons(fsr_threshold=current_fsr_threshold)
            print(f" 剪枝的FSR阈值是: {current_fsr_threshold:.4f}")
            test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device, train_time)  # 你训练里的测试函数
            pruning_started = True
            pruning_count = 1
            first_prune_epoch = epoch
            current_fsr_threshold += 0.15
            prune_epoch.append(epoch)
            # neuron_acc_history.append((num_pruned, test_acc))

            neuron_acc_history.append(
                (num_pruned, last_prune_acc if last_prune_acc is not None else test_acc, num_retained))
            last_prune_acc = None  # 重置

            print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
            print(f" 剪掉神经元数量: {num_pruned}")
            num_pruned = num_pruned
            # print(f" 剪枝后剩余神经元数量: {num_retained}")

            # add:fisher info
            fisher_np = fisher_info.detach().cpu().numpy()
            fsr_np = fsr_per_neuron.detach().cpu().numpy()

            spearman_corr_val = spearman_corr(fsr_np, fisher_np)  # 你自定义的函数
            pearson_corr_val = np.corrcoef(fsr_np, fisher_np)[0, 1]

            # 加入列表中（你需要新建两个列表）
            spearman_list.append(spearman_corr_val)
            pearson_list.append(pearson_corr_val)

        elif pruning_started and not prune_stopped:
            # if num_retained < 100:
            #     print(f" ⚠️ 剪枝后剩余神经元为 {num_retained}，少于最小要求 100，停止进一步剪枝。")
            #     prune_stopped = True

            # 如果已开始剪枝，继续每隔20个epoch剪
            if num_retained > 60:
                if (epoch - first_prune_epoch) % args.prune_period == 0:
                    #add
                    fisher_info = compute_fisher_information(net, train_data_loader, args.device)

                    num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons(fsr_threshold=current_fsr_threshold)
                    print(f" 剪枝的FSR阈值是: {current_fsr_threshold:.4f}")
                    test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,
                                                               train_time)  # 你训练里的测试函数
                    if 100 <= num_retained < 200:
                        current_fsr_threshold += 0.1
                    elif num_retained < 100:
                        current_fsr_threshold += 0.05
                    else:
                        current_fsr_threshold += 0.15
                    pruning_count += 1
                    prune_epoch.append(epoch)
                    # neuron_acc_history.append((num_pruned, test_acc))
                    neuron_acc_history.append(
                        (num_pruned, last_prune_acc if last_prune_acc is not None else test_acc, num_retained))
                    last_prune_acc = None  # 重置

                    print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
                    print(f" 剪掉神经元数量: {num_pruned}")
                    num_pruned = num_pruned
                    # print(f" 剪枝后剩余神经元数量: {num_retained}")

                    # add:fisher info
                    fisher_np = fisher_info.detach().cpu().numpy()
                    fsr_np = fsr_per_neuron.detach().cpu().numpy()

                    spearman_corr_val = spearman_corr(fsr_np, fisher_np)  # 你自定义的函数
                    pearson_corr_val = np.corrcoef(fsr_np, fisher_np)[0, 1]

                    # 加入列表中（你需要新建两个列表）
                    spearman_list.append(spearman_corr_val)
                    pearson_list.append(pearson_corr_val)
            elif 60 >= num_retained > 40:
                if (epoch - first_prune_epoch) % 30 == 0:
                    # add
                    fisher_info = compute_fisher_information(net, train_data_loader, args.device)

                    num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons_num(prune_num=10 + num_pruned)
                    test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,
                                                               train_time)  # 你训练里的测试函数
                    pruning_count += 1
                    prune_epoch.append(epoch)
                    # neuron_acc_history.append((num_pruned, test_acc))
                    neuron_acc_history.append(
                        (num_pruned, last_prune_acc if last_prune_acc is not None else test_acc, num_retained))
                    last_prune_acc = None  # 重置

                    print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
                    print(f" 剪掉神经元数量: {num_pruned}")
                    num_pruned = num_pruned
                    # print(f" 剪枝后剩余神经元数量: {num_retained}")

                    # add:fisher info
                    fisher_np = fisher_info.detach().cpu().numpy()
                    fsr_np = fsr_per_neuron.detach().cpu().numpy()

                    spearman_corr_val = spearman_corr(fsr_np, fisher_np)  # 你自定义的函数
                    pearson_corr_val = np.corrcoef(fsr_np, fisher_np)[0, 1]

                    # 加入列表中（你需要新建两个列表）
                    spearman_list.append(spearman_corr_val)
                    pearson_list.append(pearson_corr_val)



            elif 40 >= num_retained > 30:
                if (epoch - first_prune_epoch) % 30 == 0:
                    # add
                    fisher_info = compute_fisher_information(net, train_data_loader, args.device)

                    num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons_num(prune_num=5 + num_pruned)
                    test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,
                                                               train_time)  # 你训练里的测试函数
                    pruning_count += 1
                    prune_epoch.append(epoch)
                    # neuron_acc_history.append((num_pruned, test_acc))
                    neuron_acc_history.append(
                        (num_pruned, last_prune_acc if last_prune_acc is not None else test_acc, num_retained))
                    last_prune_acc = None  # 重置

                    print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
                    print(f" 剪掉神经元数量: {num_pruned}")
                    num_pruned = num_pruned
                    # print(f" 剪枝后剩余神经元数量: {num_retained}")

                    # add:fisher info
                    fisher_np = fisher_info.detach().cpu().numpy()
                    fsr_np = fsr_per_neuron.detach().cpu().numpy()

                    spearman_corr_val = spearman_corr(fsr_np, fisher_np)  # 你自定义的函数
                    pearson_corr_val = np.corrcoef(fsr_np, fisher_np)[0, 1]

                    # 加入列表中（你需要新建两个列表）
                    spearman_list.append(spearman_corr_val)
                    pearson_list.append(pearson_corr_val)

            elif 30 >= num_retained > 20:
                if (epoch - first_prune_epoch) % 30 == 0:
                    # add
                    fisher_info = compute_fisher_information(net, train_data_loader, args.device)

                    num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons_num(prune_num=3 + num_pruned)
                    test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,
                                                               train_time)  # 你训练里的测试函数
                    pruning_count += 1
                    prune_epoch.append(epoch)
                    # neuron_acc_history.append((num_pruned, test_acc))
                    neuron_acc_history.append(
                        (num_pruned, last_prune_acc if last_prune_acc is not None else test_acc, num_retained))
                    last_prune_acc = None  # 重置

                    print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
                    print(f" 剪掉神经元数量: {num_pruned}")
                    num_pruned = num_pruned
                    # print(f" 剪枝后剩余神经元数量: {num_retained}")

                    # add:fisher info
                    fisher_np = fisher_info.detach().cpu().numpy()
                    fsr_np = fsr_per_neuron.detach().cpu().numpy()

                    spearman_corr_val = spearman_corr(fsr_np, fisher_np)  # 你自定义的函数
                    pearson_corr_val = np.corrcoef(fsr_np, fisher_np)[0, 1]

                    # 加入列表中（你需要新建两个列表）
                    spearman_list.append(spearman_corr_val)
                    pearson_list.append(pearson_corr_val)

            elif num_retained <= 20:
                prune_stopped = True

                # if test_acc > prune_acc_threshold:
                #     num_retained, num_pruned = net.prune_fading_neurons(fsr_threshold=fsr_threshold)
                #     test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,train_time)  # 你训练里的测试函数
                #     pruning_count += 1
                #     prune_epoch.append(epoch)
                #     neuron_acc_history.append((num_pruned, test_acc))
                #
                #     print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
                #     print(f" 剪掉神经元数量: {num_pruned}")
                #     print(f" 剪枝后剩余神经元数量: {num_retained}")
                # else:
                #     prune_stopped = True
                #     print(f"\n--- 剪枝终止：当前 test_acc={test_acc:.4f} <= 阈值 {prune_acc_threshold} ---")

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        epoch_list.append(epoch)

    # --- 训练结束后记录最后一轮表现 ---
    neuron_acc_history.append((None, test_acc, None))

    # plt.figure()
    # plt.plot(epoch_list, train_acc_list, label=f'SNR={args.snr}')
    # # 标注剪枝发生的 epoch
    # for i, prune_ep in enumerate(prune_epoch):
    #     plt.axvline(x=prune_ep, color='red', linestyle='--', alpha=0.7)
    #     plt.text(prune_ep,
    #              max(test_acc_list) * 0.95,  # 文字位置（高度）
    #              f'Prune #{i + 1}',
    #              rotation=90,
    #              color='red',
    #              fontsize=8,
    #              ha='right',
    #              va='top')
    # plt.xlabel('Epoch')
    # plt.ylabel('Train Accuracy')
    # plt.title('Epoch vs Train Accuracy')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'train_acc_snr_{args.snr}_channel_prune.png')  # 可选，保存图像
    # plt.show()
    #
    # plt.figure()
    # plt.plot(epoch_list, test_acc_list, label=f'SNR={args.snr}')
    # # 标注剪枝发生的 epoch
    # for i, prune_ep in enumerate(prune_epoch):
    #     plt.axvline(x=prune_ep, color='red', linestyle='--', alpha=0.7)
    #     plt.text(prune_ep,
    #              max(test_acc_list) * 0.95,  # 文字位置（高度）
    #              f'Prune #{i + 1}',
    #              rotation=90,
    #              color='red',
    #              fontsize=8,
    #              ha='right',
    #              va='top')
    # plt.xlabel('Epoch')
    # plt.ylabel('Test Accuracy')
    # plt.title('Epoch vs Test Accuracy')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'test_acc_snr_{args.snr}_channel_prune.png')  # 可选，保存图像
    # plt.show()

    plt.figure()

    # 画训练准确率曲线
    plt.plot(epoch_list, train_acc_list, label=f'Train Accuracy (SNR={args.snr})', color='blue')

    # 画测试准确率曲线
    plt.plot(epoch_list, test_acc_list, label=f'Test Accuracy (SNR={args.snr})', color='black')

    plt.axhline(y=0.95, color='green', linestyle='--', linewidth=1, alpha=0.8, label='Accuracy = 0.95')
    plt.axhline(y=0.90, color='orange', linestyle='--', linewidth=1, alpha=0.8, label='Accuracy = 0.90')

    # 剪枝点：画垂直线 + 标注 + 点
    for i, prune_ep in enumerate(prune_epoch):
        plt.axvline(x=prune_ep, color='red', linestyle='--', alpha=0.7)
        plt.text(prune_ep,
                 max(max(train_acc_list), max(test_acc_list)) * 0.95,
                 f'Prune #{i + 1}',
                 rotation=90,
                 color='red',
                 fontsize=8,
                 ha='right',
                 va='top')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy (Train & Test)')
    plt.legend()  # 自动根据 label 添加图例
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'acc_snr_{args.snr}_start_fsr_{args.fsr_threshold}_lambda_{args.regular_lambda}_channel_prune_period_slow_limit_13.png')
    plt.show()

    # 拆分 neuron_acc_history 为两个列表
    # pruned_counts = [item[0] for item in neuron_acc_history]  # 横轴
    # test_accs = [item[1] for item in neuron_acc_history]  # 纵轴
    pruned_counts = [item[0] for item in neuron_acc_history[:-1]]  # 去掉最后一个，保证长度一致
    test_accs = [item[1] for item in neuron_acc_history[1:]]  # 从第二个开始，错位一位
    num_retained = [item[2] for item in neuron_acc_history[:-1]]

    plt.figure()
    plt.plot(pruned_counts, test_accs, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Neurons Pruned')
    plt.ylabel('Test Accuracy After Pruning')
    plt.title('Test Accuracy vs Pruned Neurons')
    plt.grid(True)

    # 添加文本标签
    # for x, y in zip(pruned_counts, test_accs):
    #     plt.text(x, y+0.002, f'{x}', fontsize=8, ha='left', va='bottom')  # 可调字体大小与位置

    for i, (x, y) in enumerate(zip(pruned_counts, test_accs)):
        offset = 0.002 if i % 2 == 0 else -0.004  # 偶数向上、奇数向下
        plt.text(x, y + offset, f'{x}', fontsize=8, ha='center', va='bottom' if offset > 0 else 'top')

    plt.tight_layout()
    plt.savefig(
        f'test_acc_vs_pruned_neurons_snr_{args.snr}_start_fsr_{args.fsr_threshold}_lambda_{args.regular_lambda}_period_slow_limit_13.png')
    plt.show()

    plt.figure()
    plt.plot(num_retained, test_accs, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of  Retained Neurons')
    plt.ylabel('Test Accuracy After Pruning')
    plt.title('Test Accuracy vs Retained Neurons')
    plt.grid(True)

    # for x, y in zip(num_retained, test_accs):
    #     plt.text(x, y + 0.002, f'{x}', fontsize=8, ha='left', va='bottom')  # 稍微上移避免遮挡

    for i, (x, y) in enumerate(zip(num_retained, test_accs)):
        offset = 0.002 if i % 2 == 0 else -0.004  # 偶数向上、奇数向下
        plt.text(x, y + offset, f'{x}', fontsize=8, ha='center', va='bottom' if offset > 0 else 'top')

    plt.tight_layout()
    plt.savefig(
        f'test_acc_vs_retained_neurons_snr_{args.snr}_start_fsr_{args.fsr_threshold}_lambda_{args.regular_lambda}_period_slow_limit_13.png')
    plt.show()

    # print(f"{'Step':<10}{'Pruned':<10}{'Retained':<15}{'Accuracy'}")
    # print("-" * 45)
    # for i, (p, r, acc) in enumerate(zip(pruned_counts, num_retained, test_accs), start=1):
    #     print(f"{i:<10}{p:<10}{r:<15}{acc:.4f}")


    print(f"{'Step':<10}{'Epoch':<10}{'Pruned':<10}{'Retained':<12}{'Accuracy':<10}{'Spearman':<10}{'Pearson'}")
    print("-" * 80)

    for i, (p, r, acc, prune_ep, s, pe) in enumerate(zip(
            pruned_counts, num_retained, test_accs, prune_epoch, spearman_list, pearson_list), start=1):
        print(f"{i:<10}{prune_ep:<10}{p:<10}{r:<12}{acc:<10.4f}{s:<10.4f}{pe:.4f}")


if __name__ == '__main__':
    main()

