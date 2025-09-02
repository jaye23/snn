from torch.cuda import amp
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import DVSNet_Channel_Pruning_proportion
from functions_proportion import *
import torchvision.transforms as transforms
import time
import os
import argparse
import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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



def main():
    # python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:1', help='device')
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=24, type=int, metavar='N',
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
    parser.add_argument('-prune-acc-threshold', type=float, default=0.97,
                        help='accuracy threshold to trigger pruning')
    parser.add_argument('-prune-acc-stop', type=float, default=0.60,
                        help='accuracy threshold to stop pruning')
    parser.add_argument('-prune-loss-threshold', type=float, default=0.0175,
                        help='loss threshold to trigger pruning')
    parser.add_argument('-fsr-threshold', type=float, default=0.25,
                        help='FSR threshold for pruning')  # 改一下
    parser.add_argument('-regular-lambda', type=float, default=1e-8,
                        help='regularization strength')  # 改一下1e-3  shell脚本跑多个参数，保留神经元到100以下
    parser.add_argument('-min-neurons-threshold', type=int, default=10,
                        help='minimum number of neurons to stop pruning')
    parser.add_argument('-prune-period', type=int, default=20,
                        help='pruning cycle in epochs (perform pruning every N epochs once triggered)')
    parser.add_argument('-s', type=int, default=3,
                        help='the time that train the code')
    parser.add_argument('-r', type=int, default=1,
                        help='load r_th date')

    args = parser.parse_args()
    args.opt = "adam"
    print(args)

    set_seed(2025)

    pdp_values = torch.tensor(args.pdp_values)
    PDP = pdp_values / pdp_values.sum()  # 归一化
    PDP = PDP.to(args.device)  # 确保 PDP 在正确的计算设备上




    net = DVSNet_Channel_Pruning_proportion.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode,
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

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    start_epoch = 0
    try:
        start_epoch, _ , train_acc_list, test_acc_list, epoch_list,spearman_list, pearson_list,pruning_started, prune_stopped, first_prune_epoch, pruning_count, prune_check, neuron_acc_history,prune_epoch,num_retained= load_checkpoint(net, optimizer, lr_scheduler, f'checkpoints_ratio_{args.r}/checkpoint_epoch_289.pth')
        start_epoch += 1  # 从下一个 epoch 开始

    except FileNotFoundError:
        print("No checkpoint found, training from scratch.")
        test_acc_list = []
        train_acc_list = []
        epoch_list = []
        spearman_list = []
        pearson_list = []
        pruning_count = 0
        first_prune_epoch = None  # 记录首次剪枝的 epoch
        neuron_acc_history = []
        # --- 新增：定义标志位和触发阈值 ---
        pruning_started = False  # 标记开始剪枝
        prune_stopped = False  # 一旦 True，不再剪枝
        prune_epoch = []  # 新增：用于记录prune发生的epoch，-1代表还未发生
        prune_check = 0

    print(f"Training will start from epoch {start_epoch}")




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

        lr_scheduler.step()

        # test
        test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device, train_time)
        last_prune_acc = test_acc

        # if not pruning_started and test_loss <= args.prune_loss_threshold and test_acc >= args.prune_acc_threshold:
        prune_check = 0
        try:
            num_retained
        except NameError:
            num_retained = 512
            
        if num_retained > 60:
            # if (epoch - first_prune_epoch) % args.prune_period == 0 and train_acc >= 0.98:# 如果已开始剪枝，继续每隔20个epoch剪
            if  train_acc >= 0.98:
                num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons_ratio(prune_ratio = 0.3)
                prune_check = 1
                test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,train_time)  # 你训练里的测试函数
                pruning_count, last_prune_acc = record_pruning_info(epoch, pruning_count, num_pruned, num_retained,
                                                                    last_prune_acc, test_loss, test_acc, test_speed,
                                                                    neuron_acc_history, prune_epoch)

        elif 60 >= num_retained > 40:
            # if (epoch - first_prune_epoch) % 30 == 0 and train_acc >= 0.98:
            if train_acc >= 0.98:
                num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons_num(prune_num=10 + num_pruned)
                prune_check = 1
                test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,train_time)  # 你训练里的测试函数
                pruning_count, last_prune_acc = record_pruning_info(epoch, pruning_count, num_pruned, num_retained,
                                                                    last_prune_acc, test_loss, test_acc, test_speed,
                                                                    neuron_acc_history, prune_epoch)

        elif 40 >= num_retained > 30 :
            # if (epoch - first_prune_epoch) % 30 == 0  and train_acc >= 0.98 :
            if train_acc >= 0.98:
                num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons_num(prune_num=5 + num_pruned)
                prune_check = 1
                test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,train_time)  # 你训练里的测试函数
                pruning_count, last_prune_acc = record_pruning_info(epoch, pruning_count, num_pruned, num_retained,
                                                                    last_prune_acc, test_loss, test_acc, test_speed,
                                                                    neuron_acc_history, prune_epoch)


        elif 30 >= num_retained > 11:
            # if (epoch - first_prune_epoch) % 30 == 0 and train_acc >= 0.98:
            if train_acc >= 0.98:
                num_retained, num_pruned,fsr_per_neuron = net.prune_fading_neurons_num(prune_num=3 + num_pruned)
                prune_check = 1
                test_loss, test_acc, test_speed = evaluate(net, test_data_loader, args.device,train_time)  # 你训练里的测试函数
                pruning_count, last_prune_acc = record_pruning_info(epoch, pruning_count, num_pruned, num_retained,
                                                                    last_prune_acc, test_loss, test_acc, test_speed,
                                                                    neuron_acc_history, prune_epoch)


            # elif num_retained <= 10:
            #     prune_stopped = True

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True


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


        if epoch == 150:
            save_checkpoint(net, optimizer, epoch, loss, lr_scheduler, dir_path=f'checkpoints_ratio_{args.s}/',
                            train_acc_list=train_acc_list, test_acc_list=test_acc_list, epoch_list=epoch_list,
                            spearman_list=spearman_list, pearson_list=pearson_list,pruning_started=pruning_started,
                            prune_stopped=prune_stopped,first_prune_epoch=first_prune_epoch,pruning_count=pruning_count,
                            prune_check=prune_check,neuron_acc_history=neuron_acc_history,prune_epoch=prune_epoch, num_retained= num_retained)

        elif prune_check == 1:
            save_checkpoint(net, optimizer, epoch, loss, lr_scheduler, dir_path=f'checkpoints_ratio_{args.s}/',
                            train_acc_list=train_acc_list, test_acc_list=test_acc_list, epoch_list=epoch_list,
                            spearman_list=spearman_list, pearson_list=pearson_list,pruning_started=pruning_started,
                            prune_stopped=prune_stopped,first_prune_epoch=first_prune_epoch,pruning_count=pruning_count,
                            prune_check=prune_check,neuron_acc_history=neuron_acc_history,prune_epoch=prune_epoch, num_retained= num_retained)

    # --- 训练结束后记录最后一轮表现 ---
    neuron_acc_history.append((None, test_acc, None))



    #************figure1*********
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

        if (i + 1) in [14, 15]:
            plt.text(prune_ep,
                     max(max(train_acc_list), max(test_acc_list)) * 0.95,
                     f'compressing #{i + 1}',
                     rotation=90,
                     color='red',
                     fontsize=8,
                     ha='right',
                     va='top')

    compression_line = mlines.Line2D([], [], color='red', linestyle='--', label='compressing')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch vs Accuracy (Train & Test)')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(compression_line)
    labels.append('compressing')
    plt.legend(handles, labels, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f'acc_snr_{args.snr}_start_fsr_{args.fsr_threshold}_lambda_{args.regular_lambda}_channel_prune_radio_{args.s}.png')
    plt.show()


    pruned_counts = [item[0] for item in neuron_acc_history[:-1]]  # 去掉最后一个，保证长度一致
    test_accs = [item[1] for item in neuron_acc_history[1:]]  # 从第二个开始，错位一位
    num_retained = [item[2] for item in neuron_acc_history[:-1]]

    # ************figure2*********
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
        f'test_acc_vs_pruned_neurons_snr_{args.snr}_start_fsr_{args.fsr_threshold}_lambda_{args.regular_lambda}_radio_{args.s}.png')
    plt.show()



    # ************figure 3*********
    plt.figure()
    plt.plot(num_retained, test_accs, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of  Retained Neurons')
    plt.ylabel('Test Accuracy After Pruning')
    plt.title('Test Accuracy vs Retained Neurons')
    plt.grid(True)

    for i, (x, y) in enumerate(zip(num_retained, test_accs)):
        offset = 0.002 if i % 2 == 0 else -0.004  # 偶数向上、奇数向下
        plt.text(x, y + offset, f'{x}', fontsize=8, ha='center', va='bottom' if offset > 0 else 'top')

    plt.tight_layout()
    plt.savefig(
        f'test_acc_vs_retained_neurons_snr_{args.snr}_start_fsr_{args.fsr_threshold}_lambda_{args.regular_lambda}_radio_{args.s}.png')
    plt.show()


    # ************ table *********
    # print(f"{'Step':<10}{'Epoch':<10}{'Pruned':<10}{'Retained':<12}{'Accuracy':<10}{'Spearman':<10}{'Pearson'}")
    # print("-" * 80)
    #
    # for i, (p, r, acc, prune_ep, s, pe) in enumerate(zip(
    #         pruned_counts, num_retained, test_accs, prune_epoch, spearman_list, pearson_list), start=1):
    #     print(f"{i:<10}{prune_ep:<10}{p:<10}{r:<12}{acc:<10.4f}{s:<10.4f}{pe:.4f}")

    print(f"{'Step':<10}{'Epoch':<10}{'Pruned':<10}{'Retained':<12}{'Accuracy':<10}")
    print("-" * 60)

    for i, (p, r, acc, prune_ep) in enumerate(zip(
            pruned_counts, num_retained, test_accs, prune_epoch), start=1):
        print(f"{i:<10}{prune_ep:<10}{p:<10}{r:<12}{acc:<10.4f}")


if __name__ == '__main__':
    main()

