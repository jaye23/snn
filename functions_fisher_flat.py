import torch
import random
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate, neuron
import time
import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)  # 让 NumPy 打印完整数组


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

            output_with_time, batch_spikes,fsr_per_neuron = net(frame)
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
    net.train()
    total_fisher = None
    sample_count = 0

    for frame, label in data_loader:
        frame = frame.to(device).transpose(0, 1)
        label = label.to(device)
        label_onehot = F.one_hot(label, num_classes=11).float()

        net.zero_grad()

        output_with_time, _, _ = net(frame)  # neuron_output 不再需要
        out_fr = output_with_time.mean(0)
        loss = F.mse_loss(out_fr, label_onehot)
        loss.backward()

        with torch.no_grad():
            param = net.fc.weight
            if param.grad is not None:
                grad_sq = param.grad.pow(2)
                if total_fisher is None:
                    total_fisher = grad_sq.clone()
                else:
                    total_fisher += grad_sq

        sample_count += frame.shape[1]  # batch size
        functional.reset_net(net)

    fisher_info = total_fisher / sample_count #fisher info torch.Size([512, 1024])
    # print("fisher_info:", fisher_info.shape)

    # out_importance = fisher_info.sum(dim=1)  # [512]，L1式

    U, S, Vh = torch.linalg.svd(fisher_info, full_matrices=False)
    # print("U:", U.shape)
    # print("S:", S.shape)
    # print("Vh:", Vh.shape)

    V = Vh.conj().T                              # (n x n) 右奇异向量矩阵
    in_importance = (V.pow(2) @ S.pow(2))        # (n,) 每个输入神经元/通道的 Fisher 能量


    # ========= ③ 归一化（便于与 FSR 对比/画图） =========
    denom = torch.linalg.norm(in_importance) + 1e-12
    in_importance = in_importance / denom

    return in_importance                         # 形状: [in_features]


def log_fisher_fsr_statistics(fisher_info, fsr_per_neuron,
                               epoch,  save_dir,
                               spearman_list, pearson_list):
    """
    计算 FSR 与 Fisher 信息的相关性、可视化，并记录进列表中。

    参数:
    - fisher_info: torch.Tensor, Fisher 信息
    - fsr_per_neuron: torch.Tensor, FSR 信息
    - pruned_indices: torch.Tensor 或 list, 被剪掉的神经元索引
    - epoch: 当前 epoch
    - pruning_count: 当前是第几次剪枝
    - save_dir: 保存图像的路径
    - spearman_list: 用于保存 spearman 相关系数的列表（传引用）
    - pearson_list: 用于保存 pearson 相关系数的列表（传引用）
    """
    fisher_np = fisher_info.detach().cpu().numpy()
    fsr_np = fsr_per_neuron.detach().cpu().numpy()

    # 计算相关性
    spearman_corr_val = spearman_corr(fsr_np, fisher_np)
    pearson_corr_val = np.corrcoef(fsr_np, fisher_np)[0, 1]

    print("Spearman 相关系数（FSR vs Fisher Info）:", spearman_corr_val)
    print("Pearson 相关系数（FSR vs Fisher Info）:", pearson_corr_val)

    # 加入列表中
    spearman_list.append(spearman_corr_val)
    pearson_list.append(pearson_corr_val)

    assert fisher_np.shape == fsr_np.shape, "Fisher 和 FSR 形状必须一致"

    indices = np.arange(len(fsr_np))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    width = 1

    # print("fsr^2:", fsr_np**2)
    # print("fisher:", fisher_np)

    # 第一张子图：FSR
    axes[0].bar(indices, fsr_np ** 2, width=width, label='FSR', color='skyblue')
    axes[0].set_ylabel("FSR Value")
    axes[0].set_title(f"FSR (Epoch {epoch})")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # 第二张子图：Fisher Info
    axes[1].bar(indices, fisher_np, width=width, label='Fisher Info', color='orange')
    axes[1].set_xlabel("Neuron Index")
    axes[1].set_ylabel("Fisher Value")
    axes[1].set_title(f"Fisher Information (Epoch {epoch})")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    # 在整张图底部加 Spearman / Pearson 相关系数
    fig.text(0.5, -0.02,  # (x, y) = 归一化坐标系，0.5=水平居中，-0.02=稍微往下
             f"Spearman: {spearman_corr_val:.4f}   |   Pearson: {pearson_corr_val:.4f}",
             ha='center', va='top', fontsize=14)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fisher_vs_fsr_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 可视化
    # plot_fisher_vs_fsr(fisher_np, fsr_np, epoch, save_dir=save_dir)




# def plot_fisher_vs_fsr(fisher_np, fsr_np, epoch, save_dir,):
#     """
#     绘制每轮剪枝后神经元的 FSR 与 Fisher 信息柱状图（不做零值过滤）
#
#     参数：
#     - fisher_np: np.array[N]，每个神经元的 Fisher 信息
#     - fsr_np: np.array[N]，每个神经元的 FSR
#     - epoch: 当前剪枝发生的 epoch 编号
#     - save_dir: 图像保存目录
#     """


#********修改部分******#
# def plot_fisher_vs_fsr(fisher_np, fsr_np, epoch, save_dir):
#     """
#     绘制每轮剪枝后神经元的 FSR 与 Fisher 信息柱状图（剔除值为0的神经元）
#
#     参数：
#     - fisher_np: np.array[N]，每个神经元的 Fisher 信息
#     - fsr_np: np.array[N]，每个神经元的 FSR
#     - pruned_indices: list 或 Tensor，被剪掉的神经元索引
#     - epoch: 当前剪枝发生的 epoch 编号
#     - pruning_count: 当前剪枝轮数
#     - save_dir: 图像保存目录
#     """
#     assert fisher_np.shape == fsr_np.shape, "Fisher 和 FSR 形状必须一致"
#
#     # 掩码：同时剔除 fsr 和 fisher 都为 0 的神经元
#     nonzero_mask = ~((fsr_np == 0) & (fisher_np == 0))
#     filtered_fsr = fsr_np[nonzero_mask]
#     filtered_fisher = fisher_np[nonzero_mask]
#     filtered_indices = np.arange(len(fsr_np))[nonzero_mask]
#
#     fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
#
#     width = 0.6
#
#     # 第一张子图：FSR
#     axes[0].bar(filtered_indices, filtered_fsr, width=width, label='FSR', color='skyblue')
#     axes[0].set_ylabel("FSR Value")
#     axes[0].set_title(f"FSR (Epoch {epoch})")
#     axes[0].legend()
#     axes[0].grid(True, linestyle='--', alpha=0.4)
#
#     # 第二张子图：Fisher Info
#     axes[1].bar(filtered_indices, filtered_fisher, width=width, label='Fisher Info', color='orange')
#     axes[1].set_xlabel("Neuron Index")
#     axes[1].set_ylabel("Fisher Value")
#     axes[1].set_title(f"Fisher Information (Epoch {epoch})")
#     axes[1].legend()
#     axes[1].grid(True, linestyle='--', alpha=0.4)
#
#     plt.tight_layout()
#
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"fisher_vs_fsr_epoch_{epoch}.png")
#     plt.savefig(save_path)
#     plt.close()
# #********修改部分******#

def spearman_corr(x, y):
    # 将两个向量转换为排名
    x_rank = np.argsort(np.argsort(x))
    y_rank = np.argsort(np.argsort(y))

    # 计算皮尔逊相关系数
    return np.corrcoef(x_rank, y_rank)[0, 1]
def set_seed(seed):
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # NumPy 随机数
    torch.manual_seed(seed)  # PyTorch CPU 随机数
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数（单卡）
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU 随机数（多卡）

    # cuDNN 中一些操作是非确定性的，这里强制它为确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, scheduler=None, dir_path='checkpoints/',
                    train_acc_list=None, test_acc_list=None, epoch_list=None):
    """
    精简版：仅保存基础训练信息与可选的训练/验证曲线。
    - 必选：model, optimizer, epoch, loss
    - 可选：scheduler, train/test/epoch 曲线
    """
    os.makedirs(dir_path, exist_ok=True)
    filename = f'checkpoint_epoch_{epoch}.pth'
    path = os.path.join(dir_path, filename)

    checkpoint = {
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(loss) if loss is not None else None,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # 可选训练曲线
    if train_acc_list is not None:
        checkpoint['train_acc_list'] = train_acc_list
    if test_acc_list is not None:
        checkpoint['test_acc_list'] = test_acc_list
    if epoch_list is not None:
        checkpoint['epoch_list'] = epoch_list

    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')


def load_checkpoint(model, optimizer, scheduler=None, path='checkpoints/checkpoint_epoch_150.pth',
                    map_location=None, strict=True):
    """
    精简版：仅恢复基础训练信息与可选的训练/验证曲线。
    返回: epoch, loss, train_acc_list, test_acc_list, epoch_list
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # 模型权重
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if missing:
            print(f'[load_checkpoint] Missing keys: {missing}')
        if unexpected:
            print(f'[load_checkpoint] Unexpected keys: {unexpected}')

    # 优化器与调度器
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 基本信息
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    # 可选训练曲线
    train_acc_list = checkpoint.get('train_acc_list', [])
    test_acc_list = checkpoint.get('test_acc_list', [])
    epoch_list = checkpoint.get('epoch_list', [])

    print(f'Checkpoint loaded from {path}, resuming from epoch {epoch}')
    return epoch, loss, train_acc_list, test_acc_list, epoch_list


def record_pruning_info(epoch, pruning_count, num_pruned, num_retained,
                        last_prune_acc, test_loss, test_acc, test_speed,
                        neuron_acc_history, prune_epoch):
    """
    记录当前剪枝后的信息并更新日志列表。

    参数:
    - epoch: 当前 epoch
    - pruning_count: 当前第几次剪枝
    - num_pruned: 剪掉的神经元数量
    - num_retained: 剩余神经元数量
    - last_prune_acc: 剪枝前一轮的准确率（可选）
    - test_loss, test_acc, test_speed: 当前剪枝后的测试指标
    - neuron_acc_history: 精度-剪枝记录列表（引用传入）
    - prune_epoch: 剪枝发生的 epoch 列表（引用传入）

    返回:
    - pruning_count: 剪枝次数 +1 后的值
    - last_prune_acc: 重置为 None
    """
    prune_epoch.append(epoch)
    pruning_count += 1

    # 记录历史剪枝信息
    neuron_acc_history.append(
        (num_pruned, last_prune_acc if last_prune_acc is not None else test_acc, num_retained)
    )

    print(f"\n--- 第 {pruning_count} 次剪枝已执行 (Epoch: {epoch}) ---")
    print(f" 剪掉神经元数量: {num_pruned}")
    # print(f" 剪枝后剩余神经元数量: {num_retained}")

    return pruning_count, None  # last_prune_acc 重置为 None

