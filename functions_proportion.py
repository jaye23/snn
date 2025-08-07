import torch
import random
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate, neuron
import time
import os
import matplotlib.pyplot as plt
import numpy as np

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

    U, S, Vh = torch.linalg.svd(fisher_info, full_matrices=False)

    # 最大奇异值对应的左奇异向量（shape: [512]）
    dominant_vector = abs(Vh[:, 0])

    # dominant_vector = dominant_vector / dominant_vector.norm()

    return dominant_vector

# ****************
    # # 假设 fisher_info 是 shape [512, 1024] 的 torch.Tensor
    # A = fisher_info
    # AAT = A @ A.T  # shape: [512, 512]，等价于 A A^T
    #
    # # 计算 AAT 的特征值和特征向量
    # eigenvalues, eigenvectors = torch.linalg.eigh(AAT)  # 适用于对称矩阵
    #
    # # 最大特征值及其对应的特征向量
    # max_eigenvalue, max_index = torch.max(eigenvalues, dim=0)
    # dominant_eigenvector = eigenvectors[:, max_index]  # shape: [512]

#****************
    # print('fisher info', fisher_info.shape)
    # fisher_info_per_input = fisher_info.sum(dim=1)  # shape: [512]
    # fisher_info_per_input = fisher_info_per_input * net.mask


# with torch.no_grad():
#     # 计算 Fisher 信息（平均每个样本的平方梯度）
#     batch_fisher = neuron_output.grad.pow(2).sum(dim=(0, 1))  # sum over T and B
#     if total_fisher is None:
#         total_fisher = batch_fisher
#     else:
#         total_fisher += batch_fisher
#
# sample_count += neuron_output.shape[0] * neuron_output.shape[1]
# functional.reset_net(net)  # 重置膜电位等状态，准备下一个样本




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


def save_checkpoint(model, optimizer, epoch, loss, scheduler=None, dir_path='checkpoints/',train_acc_list=None,
                    test_acc_list=None, epoch_list=None,spearman_list=None,pearson_list=None,pruning_started=None, prune_stopped=None,
                    first_prune_epoch=None, pruning_count=None,
                    prune_check=None, neuron_acc_history=None,prune_epoch=None, num_retained=None):
    os.makedirs(dir_path, exist_ok=True)

    filename = f'checkpoint_epoch_{epoch}.pth'
    path = os.path.join(dir_path, filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if train_acc_list is not None and test_acc_list is not None and epoch_list is not None and spearman_list is not None and pearson_list is not None and neuron_acc_history is not None and prune_epoch is not None:
        checkpoint['train_acc_list'] = train_acc_list
        checkpoint['test_acc_list'] = test_acc_list
        checkpoint['epoch_list'] = epoch_list
        checkpoint['spearman_list'] = spearman_list
        checkpoint['pearson_list'] = pearson_list
        checkpoint['neuron_acc_history'] = neuron_acc_history
        checkpoint['prune_epoch'] = prune_epoch

    if num_retained is not None:
        checkpoint['num_retained'] = num_retained

    checkpoint['pruning_started'] = pruning_started
    checkpoint['prune_stopped'] = prune_stopped
    checkpoint['first_prune_epoch'] = first_prune_epoch
    checkpoint['pruning_count'] = pruning_count
    checkpoint['prune_check'] = prune_check
    checkpoint['mask'] = model.mask.cpu()  # 建议保存为 cpu 版本

    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')



def load_checkpoint(model, optimizer, scheduler=None, path='checkpoints/checkpoint_epoch_150.pth'):
    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    train_acc_list = checkpoint.get('train_acc_list', [])
    test_acc_list = checkpoint.get('test_acc_list', [])
    epoch_list = checkpoint.get('epoch_list', [])
    spearman_list = checkpoint.get('spearman_list', [])
    pearson_list = checkpoint.get('pearson_list', [])

    pruning_started = checkpoint.get('pruning_started', False)
    prune_stopped = checkpoint.get('prune_stopped', False)
    first_prune_epoch = checkpoint.get('first_prune_epoch', None)
    pruning_count = checkpoint.get('pruning_count', 0)
    prune_check = checkpoint.get('prune_check', 0)
    neuron_acc_history = checkpoint.get('neuron_acc_history', [])
    prune_epoch = checkpoint.get('prune_epoch', [])
    num_retained=checkpoint.get('num_retained', 512)
    if 'mask' in checkpoint:
        model.mask = checkpoint['mask'].to(model.device)  # ✅ 恢复 mask

    print(f'Checkpoint loaded from {path}, resuming from epoch {epoch}')
    return epoch, loss ,train_acc_list, test_acc_list, epoch_list, spearman_list, pearson_list,pruning_started, prune_stopped, first_prune_epoch,pruning_count, prune_check, neuron_acc_history,prune_epoch, num_retained


def plot_fisher_vs_fsr(fisher_np, fsr_np, pruned_indices, epoch, pruning_count, save_dir):
    """
    绘制每轮剪枝后神经元的 FSR 与 Fisher 信息柱状图（剔除值为0的神经元，重排横轴）

    参数：
    - fisher_np: np.array[N]，每个神经元的 Fisher 信息
    - fsr_np: np.array[N]，每个神经元的 FSR
    - pruned_indices: list 或 Tensor，被剪掉的神经元索引
    - epoch: 当前剪枝发生的 epoch 编号
    - pruning_count: 当前剪枝轮数
    - save_dir: 图像保存目录
    """
    assert fisher_np.shape == fsr_np.shape, "Fisher 和 FSR 形状必须一致"

    # 掩码：同时剔除 fsr 和 fisher 都为 0 的神经元
    nonzero_mask = ~(fsr_np == 0)
    filtered_fsr = fsr_np[nonzero_mask]
    filtered_fisher = fisher_np[nonzero_mask]

    # 🟢 横轴编号重排为 0, 1, ..., N'-1
    filtered_indices = np.arange(len(filtered_fsr))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    width = 0.6

    # 第一张子图：FSR
    axes[0].bar(filtered_indices, filtered_fsr, width=width, label='FSR', color='skyblue')
    axes[0].set_ylabel("FSR Value")
    axes[0].set_title(f"FSR (Epoch {epoch}, # {pruning_count})")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # 第二张子图：Fisher Info
    axes[1].bar(filtered_indices, filtered_fisher, width=width, label='Fisher Info', color='orange')
    axes[1].set_xlabel("Neuron Index")
    axes[1].set_ylabel("Fisher Value")
    axes[1].set_title(f"Fisher Information (Epoch {epoch}, # {pruning_count})")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fisher_vs_fsr_epoch_{epoch}_prune_{pruning_count}.png")
    plt.savefig(save_path)
    plt.close()



# def plot_fisher_vs_fsr(fisher_np, fsr_np, pruned_indices, epoch, pruning_count, save_dir):
#     """
#     绘制每轮剪枝后神经元的 FSR 与 Fisher 信息柱状图
#
#     参数：
#     - fisher_info: Tensor[N]，每个神经元的 Fisher 信息
#     - fsr_per_neuron: Tensor[N]，每个神经元的 FSR
#     - pruned_indices: list 或 Tensor，被剪掉的神经元索引
#     - epoch: 当前剪枝发生的 epoch 编号
#     - save_dir: 图像保存目录
#     """
#     indices = np.arange(len(fsr_np))
#
#     fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
#
#     width = 0.6  # 子图中可以稍微加宽柱状图的宽度
#
#     # 第一张子图：FSR
#     axes[0].bar(indices, fsr_np, width=width, label='FSR', color='skyblue')
#     axes[0].set_ylabel("FSR Value")
#     axes[0].set_title(f"FSR (Epoch {epoch}, # {pruning_count})")
#     axes[0].legend()
#     axes[0].grid(True, linestyle='--', alpha=0.4)
#
#     # # 红色虚线标出剪掉的神经元
#     # if pruned_indices is not None and len(pruned_indices) > 0:
#     #     for idx in pruned_indices.cpu().tolist():
#     #         axes[0].axvline(x=idx, color='red', linestyle='--', linewidth=0.6, alpha=0.6)
#
#     # 第二张子图：Fisher Info
#     axes[1].bar(indices, fisher_np, width=width, label='Fisher Info', color='orange')
#     axes[1].set_xlabel("Neuron Index")
#     axes[1].set_ylabel("Fisher Value")
#     axes[1].set_title(f"Fisher Information (Epoch {epoch}, # {pruning_count})")
#     axes[1].legend()
#     axes[1].grid(True, linestyle='--', alpha=0.4)
#     #
#     # # 红色虚线标出剪掉的神经元
#     # if pruned_indices is not None and len(pruned_indices) > 0:
#     #     for idx in pruned_indices.cpu().tolist():
#     #         axes[1].axvline(x=idx, color='red', linestyle='--', linewidth=0.6, alpha=0.6)
#
#     plt.tight_layout()
#
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"fisher_vs_fsr_epoch_{epoch}_prune_{pruning_count}.png")
#     plt.savefig(save_path)
#     plt.close()

def log_fisher_fsr_statistics(fisher_info, fsr_per_neuron, pruned_indices,
                               epoch, pruning_count, save_dir,
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

    # 可视化
    plot_fisher_vs_fsr(fisher_np, fsr_np, pruned_indices, epoch, pruning_count, save_dir=save_dir)

    # 计算相关性
    spearman_corr_val = spearman_corr(fsr_np, fisher_np)
    pearson_corr_val = np.corrcoef(fsr_np, fisher_np)[0, 1]

    # 加入列表中
    spearman_list.append(spearman_corr_val)
    pearson_list.append(pearson_corr_val)


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

