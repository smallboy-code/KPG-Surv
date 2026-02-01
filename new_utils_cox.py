import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
from lifelines import KaplanMeierFitter

# 在 new_utils_cox.py 文件末尾添加

def plot_kaplan_meier_survival_curve(y_true, y_pred, y_time, epoch, save_dir,
                                     prefix='val', keep_latest_only=True, n_groups=2):


    os.makedirs(save_dir, exist_ok=True)


    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_time = np.array(y_time).flatten()


    valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_time) & ~np.isnan(y_true)
    y_pred = y_pred[valid_mask]
    y_time = y_time[valid_mask]
    y_true = y_true[valid_mask]

    if len(y_pred) < 10:
        print(f"警告: 有效数据太少 ({len(y_pred)}), 跳过KM曲线绘制")
        return None, None

    # 根据风险预测值分组
    if n_groups == 2:

        median_risk = np.median(y_pred)
        groups = (y_pred > median_risk).astype(int)
        group_labels = ['低风险组 (≤中位数)', '高风险组 (>中位数)']
    elif n_groups == 3:

        terciles = np.percentile(y_pred, [33.3, 66.7])
        groups = np.zeros_like(y_pred, dtype=int)
        groups[y_pred <= terciles[0]] = 0
        groups[(y_pred > terciles[0]) & (y_pred <= terciles[1])] = 1
        groups[y_pred > terciles[1]] = 2
        group_labels = ['低风险组 (0-33%)', '中风险组 (33-67%)', '高风险组 (67-100%)']
    elif n_groups == 4:

        quartiles = np.percentile(y_pred, [25, 50, 75])
        groups = np.zeros_like(y_pred, dtype=int)
        groups[y_pred <= quartiles[0]] = 0
        groups[(y_pred > quartiles[0]) & (y_pred <= quartiles[1])] = 1
        groups[(y_pred > quartiles[1]) & (y_pred <= quartiles[2])] = 2
        groups[y_pred > quartiles[2]] = 3
        group_labels = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
    else:
        raise ValueError("n_groups 必须为2, 3或4")


    plt.figure(figsize=(10, 8))


    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']


    kmf_dict = {}
    p_values = {}

    for i in range(n_groups):
        group_mask = (groups == i)
        if np.sum(group_mask) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(
                y_time[group_mask],
                y_true[group_mask],
                label=f'{group_labels[i]} (n={np.sum(group_mask)})'
            )
            kmf.plot_survival_function(
                ci_show=True,
                color=colors[i % len(colors)],
                linewidth=2
            )
            kmf_dict[i] = kmf

    # 计算组间统计比较
    if n_groups >= 2:

        try:
            if n_groups == 2:

                group0_mask = (groups == 0)
                group1_mask = (groups == 1)

                results = logrank_test(
                    y_time[group0_mask], y_time[group1_mask],
                    event_observed_A=y_true[group0_mask],
                    event_observed_B=y_true[group1_mask]
                )
                p_values['group0_vs_group1'] = results.p_value


                plt.title(f'Kaplan-Meier 生存曲线 - Epoch {epoch} ({prefix})\n'
                          f'Log-rank p={results.p_value:.4f}')
            else:

                plt.title(f'Kaplan-Meier 生存曲线 - Epoch {epoch} ({prefix})')


                for i in kmf_dict:
                    median_survival = kmf_dict[i].median_survival_time_
                    if median_survival is not None:
                        print(f"组 {i} 中位生存时间: {median_survival:.2f}")
        except Exception as e:
            print(f"计算log-rank检验时出错: {e}")
            plt.title(f'Kaplan-Meier 生存曲线 - Epoch {epoch} ({prefix})')
    else:
        plt.title(f'Kaplan-Meier 生存曲线 - Epoch {epoch} ({prefix})')

    # 设置标签
    plt.xlabel('生存时间 (天)', fontsize=12)
    plt.ylabel('生存概率', fontsize=12)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # 添加风险表
    plt.tight_layout()

    # 保存图片
    if keep_latest_only:
        km_path = os.path.join(save_dir, f'{prefix}_km_curve_latest.png')
        km_json_path = os.path.join(save_dir, f'{prefix}_km_data_latest.json')
    else:
        km_path = os.path.join(save_dir, f'{prefix}_km_curve_epoch_{epoch:03d}.png')
        km_json_path = os.path.join(save_dir, f'{prefix}_km_data_epoch_{epoch:03d}.json')

    plt.savefig(km_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存数据到JSON
    km_data = {
        'epoch': epoch,
        'prefix': prefix,
        'n_groups': n_groups,
        'group_labels': group_labels,
        'n_samples_per_group': {str(i): int(np.sum(groups == i)) for i in range(n_groups)},
        'p_values': p_values,
        'median_survival_times': {
            str(i): float(kmf_dict[i].median_survival_time_) if i in kmf_dict else None
            for i in range(n_groups)
        },
        'predictions': y_pred.tolist(),
        'survival_times': y_time.tolist(),
        'survival_status': y_true.tolist(),
        'groups': groups.tolist()
    }

    with open(km_json_path, 'w') as f:
        json.dump(km_data, f, indent=2)

    print(f"Kaplan-Meier曲线已保存: {km_path}")

    return km_path, km_data


def save_km_curves(y_true, y_pred, y_time, epoch, save_dir, prefix='val',
                   keep_latest_only=True, n_groups=2):

    try:
        return plot_kaplan_meier_survival_curve(
            y_true, y_pred, y_time, epoch, save_dir,
            prefix, keep_latest_only, n_groups
        )
    except Exception as e:
        print(f"保存KM曲线时出错: {e}")
        return None, None
def save_roc_curve(y_true, y_pred, epoch, save_dir, prefix='val', keep_latest_only=True):

    os.makedirs(save_dir, exist_ok=True)


    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:

        if len(y_pred.shape) == 1:
            y_pred_proba = y_pred
        else:
            y_pred_proba = y_pred[:, 0]

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Epoch {epoch} ({prefix})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)


        if keep_latest_only:
            roc_path = os.path.join(save_dir, f'{prefix}_roc_latest.png')
            roc_json_path = os.path.join(save_dir, f'{prefix}_roc_latest.json')
        else:
            roc_path = os.path.join(save_dir, f'{prefix}_roc_epoch_{epoch:03d}.png')
            roc_json_path = os.path.join(save_dir, f'{prefix}_roc_epoch_{epoch:03d}.json')

        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 保存数据到JSON
        roc_data = {
            'epoch': epoch,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(roc_auc)
        }

        with open(roc_json_path, 'w') as f:
            json.dump(roc_data, f, indent=2)

        return roc_auc
    else:

        n_classes = y_pred.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # 计算每个类的ROC曲线
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        plt.figure(figsize=(12, 10))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        for i, color in zip(range(n_classes), colors):
            if i < len(colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'Class {i} (AUC = {roc_auc[i]:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multi-class ROC Curve - Epoch {epoch} ({prefix})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # 保存图片
        if keep_latest_only:
            roc_path = os.path.join(save_dir, f'{prefix}_roc_latest.png')
            roc_json_path = os.path.join(save_dir, f'{prefix}_roc_latest.json')
        else:
            roc_path = os.path.join(save_dir, f'{prefix}_roc_epoch_{epoch:03d}.png')
            roc_json_path = os.path.join(save_dir, f'{prefix}_roc_epoch_{epoch:03d}.json')

        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 计算并保存宏观平均AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes
        mean_auc = auc(all_fpr, mean_tpr)

        # 保存数据
        roc_data = {
            'epoch': epoch,
            'macro_auc': float(mean_auc),
            'per_class_auc': {str(i): float(roc_auc[i]) for i in range(n_classes)},
            'macro_fpr': all_fpr.tolist(),
            'macro_tpr': mean_tpr.tolist()
        }

        with open(roc_json_path, 'w') as f:
            json.dump(roc_data, f, indent=2)

        return mean_auc


def save_confusion_matrix(y_true, y_pred, epoch, save_dir, prefix='val', normalize=True, keep_latest_only=True):

    os.makedirs(save_dir, exist_ok=True)


    y_true = np.array(y_true).flatten()


    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        # 二值化预测结果
        if len(y_pred.shape) > 1:
            y_pred_proba = y_pred[:, 0]
        else:
            y_pred_proba = y_pred


        threshold = np.median(y_pred_proba)
        y_pred_labels = (y_pred_proba > threshold).astype(int)
    else:
        # 多分类：取最大概率的类别
        y_pred_labels = np.argmax(y_pred, axis=1)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_labels)

    # 计算准确率
    accuracy = np.trace(cm) / np.sum(cm)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)

    # 获取类别标签
    classes = np.unique(np.concatenate([y_true, y_pred_labels]))

    # 创建热图
    ax = sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                     cbar=True, square=True,
                     xticklabels=classes, yticklabels=classes)

    # 设置标签
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix - Epoch {epoch} ({prefix})\nAccuracy: {accuracy:.3f}', fontsize=16)


    if keep_latest_only:
        cm_path = os.path.join(save_dir, f'{prefix}_cm_latest.png')
        cm_json_path = os.path.join(save_dir, f'{prefix}_cm_latest.json')
    else:
        cm_path = os.path.join(save_dir, f'{prefix}_cm_epoch_{epoch:03d}.png')
        cm_json_path = os.path.join(save_dir, f'{prefix}_cm_epoch_{epoch:03d}.json')

    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存数据到JSON
    cm_data = {
        'epoch': epoch,
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'normalized': normalize,
        'predicted_labels': y_pred_labels.tolist(),
        'true_labels': y_true.tolist()
    }

    with open(cm_json_path, 'w') as f:
        json.dump(cm_data, f, indent=2)

    return accuracy, cm

def CIndex(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]:
                        concord = concord + 1
                    elif hazards[j] < hazards[i]:
                        concord = concord + 0.5
    return (concord / total)


def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy().reshape(-1)
    hazards = hazards.cpu().numpy().reshape(-1)
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)

    return (concordance_index(new_surv, -new_hazard, new_label))


def accuracy_cox(hazards, labels):
    # This accuracy is based on estimated survival events against true survival events
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    labels = labels.data.cpu().numpy()
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)


class Regularization(object):
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class NegativeLogLikelihood(nn.Module):
    def __init__(self, l2_reg):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = l2_reg
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, survtime, censor, model):
        mask = torch.ones(survtime.shape[0], survtime.shape[0])
        mask[(survtime.T - survtime) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * censor) / torch.sum(censor)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss


def CoxLoss(survtime, censor, hazard_pred, loss='cox-nnet', model=None, l2_reg=1e-2):
    if loss == 'deepsurv':
        nll_loss = NegativeLogLikelihood(l2_reg)
        return nll_loss(hazard_pred, survtime, censor, model)
    elif loss == 'cox-nnet':
        current_batch_len = len(survtime)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                if (survtime[j] >= survtime[i]): R_mat[i, j] = 1

        R_mat = torch.FloatTensor(R_mat)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        R_mat = R_mat.cuda()
        exp_theta = exp_theta.cuda()
        censor = censor.cuda()
        theta = theta.cuda()
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
        return loss_cox


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)


    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    flower_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:

        plt.bar(range(len(flower_class)), every_class_num, align='center')

        plt.xticks(range(len(flower_class)), flower_class)

        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')

        plt.xlabel('image class')

        plt.ylabel('number of images')

        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, topK, criterion, optimizer, data_loader, epoch,
                    reg_loss=False, train_flag=0, contrastive_loss_flag=0,
                    clinical_dim=0, hypoxia_dim=0, use_context_tokens=True,save_metrics_dir=None):


    model.train()
    accu_loss = torch.zeros(1).cuda()
    sample_num = 0
    pred_all = None
    survtime_torch = None
    fustat_torch = None

    total_grad_norm = 0.0
    grad_max_values = []
    grad_statistics = {}
    batch_count = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    all_preds_for_roc = []
    all_labels_for_roc = []
    all_probs_for_roc = []


    print(f"\n[DEBUG] 检查模型参数是否包含NaN...")
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)

            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
            print(f"  修复参数 {name} 中的NaN")

    if nan_params:
        print(f"  发现并修复了 {len(nan_params)} 个包含NaN的参数")


    for step, data in enumerate(data_loader):
        optimizer.zero_grad()


        if clinical_dim > 0 or hypoxia_dim > 0:

            if len(data) == 6:  # wsi, gene, clinical, hypoxia, futime, fustat
                wsi_features, gene_features, clinical_data, hypoxia_data, futime, fustat = data
            else:

                wsi_features = data[0]
                gene_features = data[1]
                futime = data[-2]
                fustat = data[-1]


                if len(data) >= 4:
                    clinical_data = data[2] if len(data[2].shape) > 0 else None
                else:
                    clinical_data = None

                if len(data) >= 5:
                    hypoxia_data = data[3] if len(data[3].shape) > 0 else None
                else:
                    hypoxia_data = None
        else:

            if len(data) == 4:
                wsi_features, gene_features, futime, fustat = data
                clinical_data, hypoxia_data = None, None
            elif len(data) == 6:
                wsi_features, gene_features, _, _, futime, fustat = data
                clinical_data, hypoxia_data = None, None
            else:
                raise ValueError(f"Unexpected data length: {len(data)}")

        sample_num += gene_features.shape[0]
        labels = torch.zeros(gene_features.shape[0], wsi_features.shape[1], gene_features.shape[1])
        labels[:, :topK, :] = 1
        labels = labels.cuda()


        if clinical_data is not None:

            if clinical_data.shape[-1] == 0:
                clinical_data = None
            else:
                clinical_data = clinical_data.float().cuda()

        if hypoxia_data is not None:

            if hypoxia_data.shape[-1] == 0:
                hypoxia_data = None
            else:
                hypoxia_data = hypoxia_data.float().cuda()


        if train_flag == 0:
            if contrastive_loss_flag == 1:

                if clinical_data is not None or hypoxia_data is not None:

                    outputs = model(wsi_features.cuda(), gene_features.cuda(), clinical_data, hypoxia_data)
                    if len(outputs) == 4:
                        gene2wsi_feature, pred, fusion_attn, wsi_attn_weights = outputs
                    else:
                        gene2wsi_feature, pred = outputs[:2]
                else:

                    outputs = model(wsi_features.cuda(), gene_features.cuda())
                    if len(outputs) == 4:
                        gene2wsi_feature, pred, fusion_attn, wsi_attn_weights = outputs
                    else:
                        gene2wsi_feature, pred = outputs[:2]

                # 计算对比损失
                if gene2wsi_feature is not None:
                    gene2wsi_feature = gene2wsi_feature.transpose(-2, -1)
                    sorted_gen2wsi_feat, _ = torch.sort(gene2wsi_feature, descending=True, dim=1)
                    gene2wsiloss = criterion(sorted_gen2wsi_feat, labels)
                else:
                    gene2wsiloss = 0

            else:

                if clinical_data is not None or hypoxia_data is not None:

                    outputs = model(wsi_features.cuda(), gene_features.cuda(), clinical_data, hypoxia_data)
                    if len(outputs) == 4:
                        pred = outputs[1]
                    else:
                        pred = outputs[0]
                else:

                    outputs = model(wsi_features.cuda(), gene_features.cuda())
                    if len(outputs) == 4:
                        pred = outputs[1]
                    else:
                        pred = outputs[0]
                gene2wsiloss = 0

        elif train_flag == 1:

            pred = model(wsi_features.cuda())
            gene2wsiloss = 0
        elif train_flag == 2:

            pred = model(gene_features.cuda())
            gene2wsiloss = 0


        # 收集预测结果用于ROC/混淆矩阵
        if pred is not None:
            all_preds_for_roc.append(pred.detach().cpu().numpy())
            all_labels_for_roc.append(fustat.cpu().numpy())

            # 对于二分类任务，收集概率值
            if pred.shape[1] == 1 or len(pred.shape) == 1:
                if len(pred.shape) == 1:
                    probs = torch.sigmoid(pred).detach().cpu().numpy()
                else:
                    probs = torch.sigmoid(pred[:, 0]).detach().cpu().numpy()
                all_probs_for_roc.append(probs)



        # 收集生存数据用于计算指标
        if step == 0:
            pred_all = pred
            survtime_torch = futime
            fustat_torch = fustat
        else:
            fustat_torch = torch.cat([fustat_torch, fustat])
            pred_all = torch.cat([pred_all, pred])
            survtime_torch = torch.cat([survtime_torch, futime])

        # 计算总损失
        futime = futime.cuda()
        fustat = fustat.cuda()

        if reg_loss and (train_flag == 0):
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss + reg_loss(model)
        elif reg_loss and (train_flag == 1):
            loss = CoxLoss(futime, fustat, pred) + reg_loss(model)
        elif not reg_loss and (train_flag == 0):
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss
        elif not reg_loss and (train_flag == 1):
            loss = CoxLoss(futime, fustat, pred)


            if torch.isnan(loss).any():
                print(f"\n[严重警告] 第{epoch}轮, 批次{step}: 损失为NaN!")
                print(f"  跳过这个批次，尝试修复...")

                # 检查哪些输入导致NaN
                print(f"  检查输入:")
                print(f"    WSI特征范围: [{wsi_features.min():.4f}, {wsi_features.max():.4f}]")
                print(f"    基因特征范围: [{gene_features.min():.4f}, {gene_features.max():.4f}]")
                if clinical_data is not None:
                    print(f"    临床特征范围: [{clinical_data.min():.4f}, {clinical_data.max():.4f}]")
                if hypoxia_data is not None:
                    print(f"    缺氧特征范围: [{hypoxia_data.min():.4f}, {hypoxia_data.max():.4f}]")


                optimizer.zero_grad()
                continue

        loss = loss.mean()
        loss.backward()



        nan_grad_layers = []
        inf_grad_layers = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad

                # 检查NaN和Inf
                if torch.isnan(grad).any():
                    nan_grad_layers.append(name)

                    param.grad.data = torch.zeros_like(param.grad.data)
                    print(f"[梯度修复] 将 {name} 的NaN梯度置零")

                if torch.isinf(grad).any():
                    inf_grad_layers.append(name)
                    # 修复Inf梯度
                    param.grad.data = torch.clamp(param.grad.data, -1e3, 1e3)
                    print(f"[梯度修复] 将 {name} 的Inf梯度裁剪")


        if nan_grad_layers:
            print(f"\n[梯度警告] 第{epoch}轮, 批次{step}: 修复了 {len(nan_grad_layers)} 个NaN梯度层")
            if len(nan_grad_layers) <= 10:
                for layer in nan_grad_layers:
                    print(f"  - {layer}")


        grad_clipped = False
        clipped_layers = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_max = param.grad.abs().max().item()


                if grad_max > 10.0:

                    torch.nn.utils.clip_grad_value_(param, 5.0)

                    clipped_layers.append(name)
                    grad_clipped = True

                    new_grad_max = param.grad.abs().max().item()
                    print(f"[梯度裁剪] 第{epoch}轮, 批次{step}: {name} - 梯度最大 {grad_max:.2f} -> {new_grad_max:.2f}")

        if grad_clipped:
            print(f"\n[梯度裁剪摘要] 第{epoch}轮, 批次{step}: 裁剪了 {len(clipped_layers)} 层的梯度")


        optimizer.step()

        accu_loss += loss
        data_loader.desc = "[train epoch {}] loss: {:.6f}".format(epoch, accu_loss.item() / (step + 1))

        if batch_count > 0:
            # 计算平均梯度范数
            avg_grad_norm = np.sqrt(total_grad_norm / batch_count)
            max_grad_value = max(grad_max_values) if grad_max_values else 0

            print(f"\n{'=' * 50}")
            print(f"第 {epoch} 轮梯度统计:")
            print(f"{'-' * 50}")
            print(f"平均梯度范数: {avg_grad_norm:.6f}")
            print(f"最大梯度值: {max_grad_value:.6f}")
            print(f"梯度值范围: [{min(grad_max_values):.6f}, {max_grad_value:.6f}]")

            # 输出分类头的梯度信息
            print(f"\n分类头梯度信息:")
            for name, stats in grad_statistics.items():
                if stats['values']:  # 确保有数据
                    avg_val = np.mean(stats['values'])
                    avg_norm = np.mean(stats['norms'])
                    print(f"  {name}: 均值={avg_val:.6f}, 范数={avg_norm:.6f}")


    if save_metrics_dir and len(all_preds_for_roc) > 0:
        os.makedirs(save_metrics_dir, exist_ok=True)


        preds_concat = np.concatenate(all_preds_for_roc, axis=0)
        labels_concat = np.concatenate(all_labels_for_roc, axis=0)


        if len(all_probs_for_roc) > 0:
            probs_concat = np.concatenate(all_probs_for_roc, axis=0)

            # 保存ROC曲线
            try:
                train_roc_auc = save_roc_curve(
                    labels_concat,
                    probs_concat,
                    epoch,
                    save_metrics_dir,
                    prefix='train',
                    keep_latest_only=True
                )
                print(f"训练ROC曲线已保存，AUC: {train_roc_auc:.4f}")
            except Exception as e:
                print(f"保存训练ROC曲线失败: {e}")

            # 保存混淆矩阵
            try:
                train_accuracy, train_cm = save_confusion_matrix(
                    labels_concat,
                    probs_concat,
                    epoch,
                    save_metrics_dir,
                    prefix='train',
                    keep_latest_only=True
                )
                print(f"训练混淆矩阵已保存，准确率: {train_accuracy:.4f}")
            except Exception as e:
                print(f"保存训练混淆矩阵失败: {e}")


        # 计算训练指标
        if pred_all is not None and fustat_torch is not None and survtime_torch is not None:
            acc = accuracy_cox(pred_all.data, fustat_torch)
            pvalue_pred = cox_log_rank(pred_all.data, fustat_torch, survtime_torch)
            c_index = CIndex_lifeline(pred_all.data, fustat_torch, survtime_torch)

            print(f"\nTraining Results - Loss: {accu_loss.item() / (step + 1):.4f}, "
                  f"C-index: {c_index:.4f}, P-value: {pvalue_pred:.4f}, Cox Acc: {acc:.4f}")


            all_preds_np = pred_all.detach().cpu().numpy()
            all_labels_np = fustat_torch.cpu().numpy()


            if all_preds_np.shape[1] == 1 or len(all_preds_np.shape) == 1:
                if len(all_preds_np.shape) == 1:
                    all_probs_np = torch.sigmoid(pred_all).detach().cpu().numpy()
                else:
                    all_probs_np = torch.sigmoid(pred_all[:, 0]).detach().cpu().numpy()
            else:
                import torch.nn.functional as F
                all_probs_np = F.softmax(pred_all, dim=1).detach().cpu().numpy()

            if all_preds_np is not None and all_labels_np is not None:
                try:

                    km_path, km_data = plot_kaplan_meier_survival_curve(
                        all_labels_np,
                        all_preds_np.flatten(),
                        survtime_torch.cpu().numpy(),
                        epoch,
                        save_metrics_dir,
                        prefix='train',
                        keep_latest_only=True,
                        n_groups=2
                    )
                    if km_path:
                        print(f"训练集KM曲线已保存: {km_path}")
                except Exception as e:
                    print(f"保存训练集KM曲线失败: {e}")

            return (accu_loss.item() / (step + 1), acc, pvalue_pred, c_index,
                    all_preds_np, all_labels_np, all_probs_np)
        else:
            print("Warning: No predictions generated during training")
            return (accu_loss.item() / (step + 1), 0.0, 1.0, 0.5,
                    None, None, None)

@torch.no_grad()
def evaluate(model, topK, criterion, data_loader, epoch, json_path, reg_loss=False, train_flag=0,
             contrastive_loss_flag=0, clinical_dim=0, hypoxia_dim=0, use_context_tokens=True,
             save_metrics_dir=None):


    model.eval()
    accu_loss = torch.zeros(1).cuda()

    sample_num = 0
    pred_all = None
    survtime_all = []
    status_all = []
    survtime_torch = None
    fustat_torch = None
    all_preds_for_roc = []
    all_labels_for_roc = []
    all_probs_for_roc = []

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        if clinical_dim > 0 or hypoxia_dim > 0:

            if len(data) == 6:  # wsi, gene, clinical, hypoxia, futime, fustat
                wsi_features, gene_features, clinical_data, hypoxia_data, futime, fustat = data
            else:

                wsi_features = data[0]
                gene_features = data[1]
                futime = data[-2]
                fustat = data[-1]


                if len(data) >= 4:
                    clinical_data = data[2] if len(data[2].shape) > 0 else None
                else:
                    clinical_data = None

                if len(data) >= 5:
                    hypoxia_data = data[3] if len(data[3].shape) > 0 else None
                else:
                    hypoxia_data = None
        else:

            if len(data) == 4:
                wsi_features, gene_features, futime, fustat = data
                clinical_data, hypoxia_data = None, None
            elif len(data) == 6:
                wsi_features, gene_features, _, _, futime, fustat = data
                clinical_data, hypoxia_data = None, None
            else:
                raise ValueError(f"Unexpected data length: {len(data)}")

        sample_num += gene_features.shape[0]
        labels = torch.zeros(gene_features.shape[0], wsi_features.shape[1], gene_features.shape[1])
        labels[:, :topK, :] = 1
        labels = labels.cuda()


        if clinical_data is not None:

            if clinical_data.shape[-1] == 0:
                clinical_data = None
            else:
                clinical_data = clinical_data.float().cuda()

        if hypoxia_data is not None:

            if hypoxia_data.shape[-1] == 0:
                hypoxia_data = None
            else:
                hypoxia_data = hypoxia_data.float().cuda()


        if train_flag == 0:
            if contrastive_loss_flag == 1:

                if clinical_data is not None or hypoxia_data is not None:

                    outputs = model(wsi_features.cuda(), gene_features.cuda(), clinical_data, hypoxia_data)
                    if len(outputs) == 4:
                        gene2wsi_feature, pred, fusion_attn, wsi_attn_weights = outputs
                    else:
                        gene2wsi_feature, pred = outputs[:2]
                else:

                    outputs = model(wsi_features.cuda(), gene_features.cuda())
                    if len(outputs) == 4:
                        gene2wsi_feature, pred, fusion_attn, wsi_attn_weights = outputs
                    else:
                        gene2wsi_feature, pred = outputs[:2]

                # 计算对比损失
                if gene2wsi_feature is not None:
                    gene2wsi_feature = gene2wsi_feature.transpose(-2, -1)
                    sorted_gen2wsi_feat, _ = torch.sort(gene2wsi_feature, descending=True, dim=1)
                    gene2wsiloss = criterion(sorted_gen2wsi_feat, labels)
                else:
                    gene2wsiloss = 0

            else:

                if clinical_data is not None or hypoxia_data is not None:

                    outputs = model(wsi_features.cuda(), gene_features.cuda(), clinical_data, hypoxia_data)
                    if len(outputs) == 4:
                        pred = outputs[1]
                    else:
                        pred = outputs[0]
                else:

                    outputs = model(wsi_features.cuda(), gene_features.cuda())
                    if len(outputs) == 4:
                        pred = outputs[1]
                    else:
                        pred = outputs[0]
                gene2wsiloss = 0

        elif train_flag == 1:

            pred = model(wsi_features.cuda())
            gene2wsiloss = 0
        elif train_flag == 2:

            pred = model(gene_features.cuda())
            gene2wsiloss = 0


        if pred is not None:
            all_preds_for_roc.append(pred.detach().cpu().numpy())
            all_labels_for_roc.append(fustat.cpu().numpy())


            if pred.shape[1] == 1 or len(pred.shape) == 1:
                if len(pred.shape) == 1:
                    probs = torch.sigmoid(pred).detach().cpu().numpy()
                else:
                    probs = torch.sigmoid(pred[:, 0]).detach().cpu().numpy()
                all_probs_for_roc.append(probs)

        # 收集生存数据
        survtime_all.append(np.squeeze(futime.data.cpu().numpy()))
        status_all.append(np.squeeze(fustat.data.cpu().numpy()))

        if step == 0:
            pred_all = pred
            survtime_torch = futime
            fustat_torch = fustat
        else:
            fustat_torch = torch.cat([fustat_torch, fustat])
            pred_all = torch.cat([pred_all, pred])
            survtime_torch = torch.cat([survtime_torch, futime])

        # 计算总损失
        futime = futime.cuda()
        fustat = fustat.cuda()

        if reg_loss and (train_flag == 0):
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss + reg_loss(model)
        elif reg_loss and (train_flag == 1):
            loss = CoxLoss(futime, fustat, pred) + reg_loss(model)
        elif not reg_loss and (train_flag == 0):
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss
        elif not reg_loss and (train_flag == 1):
            loss = CoxLoss(futime, fustat, pred)

        loss = loss.mean()
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.6f}".format(epoch, accu_loss.item() / (step + 1))


    if save_metrics_dir and len(all_preds_for_roc) > 0:
        os.makedirs(save_metrics_dir, exist_ok=True)


        preds_concat = np.concatenate(all_preds_for_roc, axis=0)
        labels_concat = np.concatenate(all_labels_for_roc, axis=0)


        if len(all_probs_for_roc) > 0:
            probs_concat = np.concatenate(all_probs_for_roc, axis=0)

            # 保存ROC曲线
            try:
                val_roc_auc = save_roc_curve(
                    labels_concat,
                    probs_concat,
                    epoch,
                    save_metrics_dir,
                    prefix='val',
                    keep_latest_only=True
                )
                print(f"验证ROC曲线已保存，AUC: {val_roc_auc:.4f}")
            except Exception as e:
                print(f"保存验证ROC曲线失败: {e}")

            # 保存混淆矩阵
            try:
                val_accuracy, val_cm = save_confusion_matrix(
                    labels_concat,
                    probs_concat,
                    epoch,
                    save_metrics_dir,
                    prefix='val',
                    keep_latest_only=True
                )
                print(f"验证混淆矩阵已保存，准确率: {val_accuracy:.4f}")
            except Exception as e:
                print(f"保存验证混淆矩阵失败: {e}")

    # 计算评估指标
    if pred_all is not None and fustat_torch is not None and survtime_torch is not None:
        acc = accuracy_cox(pred_all.data, fustat_torch)
        pvalue_pred = cox_log_rank(pred_all.data, fustat_torch, survtime_torch)
        c_index = CIndex_lifeline(pred_all.data, fustat_torch, survtime_torch)

        print(f"\nValidation Results - Loss: {accu_loss.item() / (step + 1):.4f}, "
              f"C-index: {c_index:.4f}, P-value: {pvalue_pred:.4f}, Cox Acc: {acc:.4f}")

        # 收集预测数据用于ROC
        all_preds_np = pred_all.detach().cpu().numpy()
        all_labels_np = fustat_torch.cpu().numpy()

        # 计算概率
        if all_preds_np.shape[1] == 1 or len(all_preds_np.shape) == 1:
            if len(all_preds_np.shape) == 1:
                all_probs_np = torch.sigmoid(pred_all).detach().cpu().numpy()
            else:
                all_probs_np = torch.sigmoid(pred_all[:, 0]).detach().cpu().numpy()
        else:
            import torch.nn.functional as F
            all_probs_np = F.softmax(pred_all, dim=1).detach().cpu().numpy()
        if all_preds_np is not None and all_labels_np is not None:
            try:
                km_path, km_data = plot_kaplan_meier_survival_curve(
                    all_labels_np,
                    all_preds_np.flatten(),
                    survtime_torch.cpu().numpy(),
                    epoch,
                    save_metrics_dir,
                    prefix='val',
                    keep_latest_only=True,
                    n_groups=2
                )
                if km_path:
                    print(f"验证集KM曲线已保存: {km_path}")
            except Exception as e:
                print(f"保存验证集KM曲线失败: {e}")

        return (accu_loss.item() / (step + 1), acc, pvalue_pred, c_index,
                all_preds_np, all_labels_np, all_probs_np)
    else:
        print("Warning: No predictions generated during validation")
        return (0.0, 0.0, 1.0, 0.5, None, None, None)