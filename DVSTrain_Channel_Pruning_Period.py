import torch
import random
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import DVSNet_Channel_Pruning_Period
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import time
import os
import argparse
import datetime
import matplotlib.pyplot as plt
# from IPython.display import Image, display


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


def main():
    # python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=128, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default=os.path.join(current_dir, "datasets", "DVS"), type=str, help='root dir of DVS Gesture dataset')
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
    parser.add_argument('-prune-acc-threshold', type=float, default=0.80,
                        help='accuracy threshold to trigger pruning')
    parser.add_argument('-prune-acc-stop', type=float, default=0.60,
                        help='accuracy threshold to stop pruning')
    parser.add_argument('-prune-loss-threshold', type=float, default=0.0175,
                        help='loss threshold to trigger pruning')
    parser.add_argument('-fsr-threshold', type=float, default=0.07,
                        help='FSR threshold for pruning')
    parser.add_argument('-regular-lambda', type=float, default=1e-4,
                        help='regularization strength')
    parser.add_argument('-min-neurons-threshold', type=int, default=10,
                        help='minimum number of neurons to stop pruning')


    args = parser.parse_args()
    args.opt="adam"
    print(args)

    # pdp_values = torch.tensor([1.0, 0.5, 0.3, 0.2, 0.1])  # 自定义 PDP
    PDP = args.pdp_values / args.pdp_values.sum()  # 归一化
    PDP = PDP.to(args.device)  # 确保 PDP 在正确的计算设备上

    # --- 新增：定义标志位和触发阈值 ---
    pruning_started = False #标记开始剪枝
    prune_stopped = False  # 一旦 True，不再剪枝
    # prune_acc_threshold = 0.80  # 触发剪枝的准确率阈值
    # prune_acc_stop = 0.60  # 停止剪枝的准确率阈值
    # prune_loss_threshold = 0.0175  # 触发剪枝损失阈值
    # fsr_threshold = 0.07  # FSR阈值
    prune_epoch = []  # 新增：用于记录prune发生的epoch，-1代表还未发生
    neuron_acc_history = []
    # regular_lambda = 1e-4  # 新增：正则化强度超参数
    # min_neurons_threshold = 10  # 当保留的神经元数量小于或等于这个值时，将不再进行剪枝。


    net = DVSNet_Channel_Pruning_Period.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True,PDP=PDP,snr=args.snr,device=args.device)

    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    #print(net)

    net.to(args.device)

    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number',transform = transforms.Compose([toTensor(),RandomTranslate(max_offset=25)]))
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')



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
                    output_with_time, batch_spikes = net(frame)
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
                output_with_time, batch_spikes = net(frame)
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

        if not pruning_started and test_loss <= args.prune_loss_threshold and test_acc >= args.prune_acc_threshold:
            # 启动第一次剪枝
            num_retained, num_pruned = net.prune_fading_neurons(fsr_threshold=args.fsr_threshold)
            test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device, train_time)  # 你训练里的测试函数
            pruning_started = True
            pruning_count = 1
            first_prune_epoch=epoch
            prune_epoch.append(epoch)
            neuron_acc_history.append((num_pruned, test_acc))

            print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
            print(f" 剪掉神经元数量: {num_pruned}")
            print(f" 剪枝后剩余神经元数量: {num_retained}")

        elif pruning_started and not prune_stopped:
            # 如果已开始剪枝，继续每隔20个epoch剪
            if (epoch - first_prune_epoch) % 20 == 0:
                num_retained, num_pruned = net.prune_fading_neurons(fsr_threshold=args.fsr_threshold)
                test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device, train_time)  # 你训练里的测试函数
                pruning_count += 1
                prune_epoch.append(epoch)
                neuron_acc_history.append((num_pruned, test_acc))

                print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
                print(f" 剪掉神经元数量: {num_pruned}")
                print(f" 剪枝后剩余神经元数量: {num_retained}")


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
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        epoch_list.append(epoch)

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
    plt.savefig(f'acc_snr_{args.snr}_channel_prune_combined.png')
    plt.show()

    # 拆分 neuron_acc_history 为两个列表
    pruned_counts = [item[0] for item in neuron_acc_history]  # 横轴
    test_accs = [item[1] for item in neuron_acc_history]  # 纵轴

    plt.figure()
    plt.plot(pruned_counts, test_accs, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Neurons Pruned')
    plt.ylabel('Test Accuracy After Pruning')
    plt.title('Test Accuracy vs Pruned Neurons')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'test_acc_vs_pruned_neurons_snr_{args.snr}.png')
    plt.show()




if __name__ == '__main__':
    main()

