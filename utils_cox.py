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
import time
import numpy as np
from lifelines.utils import concordance_index
import pandas as pd

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import pandas as pd


def calculate_multi_time_metrics_all_in_one(predictions, times, events,
                                            time_points=[365, 1095, 1825],
                                            bootstrap_n: int = 0,
                                            random_seed: int = 42):

    predictions_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    times_np = times.cpu().numpy() if torch.is_tensor(times) else times
    events_np = events.cpu().numpy() if torch.is_tensor(events) else events

    print(f"\n=== 多时间点指标计算 ===")
    print(f"预测形状: {predictions_np.shape}")
    print(f"样本数: {len(times_np)}")
    print(f"事件数: {np.sum(events_np)}")

    metrics = {}


    for i, t in enumerate(time_points):
        year = t // 365
        pred_probs = 1 / (1 + np.exp(-predictions_np[:, i]))

        print(f"\n{year}年生存预测:")
        print(f"  概率范围: [{pred_probs.min():.3f}, {pred_probs.max():.3f}]")
        print(f"  中位数: {np.median(pred_probs):.3f}")


        c_index = calculate_time_dependent_cindex_single(pred_probs, times_np, events_np, t)


        brier = calculate_brier_score_single(pred_probs, times_np, events_np, t)


        auc = calculate_time_specific_auc(pred_probs, times_np, events_np, t)


        accuracy = calculate_time_accuracy(pred_probs, times_np, events_np, t)


        hr, hr_ci_low, hr_ci_high = calculate_hazard_ratio_single(pred_probs, times_np, events_np, t)


        p_value = calculate_logrank_p_value(pred_probs, times_np, events_np, t)

        metrics[f'{year}年'] = {
            'c_index': float(c_index),
            'brier_score': float(brier),
            'auc': float(auc),
            'accuracy': float(accuracy),
            'hr': {
                'value': float(hr) if hr else None,
                'ci_lower': float(hr_ci_low) if hr_ci_low else None,
                'ci_upper': float(hr_ci_high) if hr_ci_high else None
            } if hr else None,
            'logrank_p': float(p_value) if p_value else None,
            'n_samples': int(np.sum((events_np == 1) | (times_np >= t))),
            'event_rate': float(np.sum((events_np == 1) & (times_np <= t)) / len(times_np)) if len(times_np) > 0 else 0
        }

        print(f"  C-index: {c_index:.4f}")
        print(f"  Brier评分: {brier:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  准确率: {accuracy:.4f}")
        if hr:
            print(f"  HR: {hr:.4f} ({hr_ci_low:.4f}-{hr_ci_high:.4f})")
        if p_value:
            print(f"  Log-rank p值: {p_value:.6f}")


    overall_metrics = calculate_overall_metrics(predictions_np, times_np, events_np)
    metrics['综合指标'] = overall_metrics

    print(f"\n=== 综合指标 ===")
    print(f"综合C指数: {overall_metrics['综合C指数']:.4f}")
    print(f"综合Brier评分: {overall_metrics['综合Brier评分']:.4f}")
    print(f"时间AUC: {overall_metrics['时间AUC']:.4f}")


    if bootstrap_n > 0:
        print(f"\n{'=' * 60}")
        print(f"使用Bootstrap计算置信区间 (n={bootstrap_n})")
        print(f"{'=' * 60}")

        try:

            import sys
            import importlib.util

            spec = importlib.util.find_spec("bootstrap_utils")
            if spec is None:
                print("[INFO] bootstrap_utils未找到，跳过Bootstrap计算")
            else:
                from bootstrap_utils import BootstrapEvaluator

                bootstrap_evaluator = BootstrapEvaluator(
                    n_bootstrap=bootstrap_n,
                    random_seed=random_seed
                )

                bootstrap_results = bootstrap_evaluator.evaluate_with_bootstrap(
                    predictions=predictions_np,
                    times=times_np,
                    events=events_np,
                    metric_function=lambda preds, t, e, tp: calculate_base_multi_time_metrics(preds, t, e, tp),
                    time_points=time_points,
                    verbose=True
                )

                if bootstrap_results is not None:
                    try:

                        final_results = merge_bootstrap_results(metrics, bootstrap_results)


                        summary = bootstrap_evaluator.bootstrap_summary_statistics(final_results)

                        print(f"\n✓ Bootstrap完成!")

                        return {
                            'metrics': final_results,
                            'bootstrap_summary': summary,
                            'bootstrap_params': {
                                'n_bootstrap': bootstrap_n,
                                'confidence_level': 0.95,
                                'random_seed': random_seed
                            }
                        }
                    except Exception as e:
                        print(f"[WARNING] 处理Bootstrap结果时出错: {str(e)[:200]}")

                        return metrics
                else:
                    print("[WARNING] Bootstrap返回None结果")
                    return metrics

        except ImportError as e:
            print(f"[INFO] bootstrap_utils导入失败: {e}")
            print("返回基础指标")
            return metrics
        except Exception as e:
            print(f"[WARNING] Bootstrap计算失败: {str(e)[:200]}")
            print("返回未加置信区间的结果")

            return metrics


def calculate_base_multi_time_metrics(predictions_np, times_np, events_np, time_points):

    base_metrics = {}

    for i, t in enumerate(time_points):
        year = t // 365
        pred_probs = 1 / (1 + np.exp(-predictions_np[:, i]))

        c_index = calculate_time_dependent_cindex_single(pred_probs, times_np, events_np, t)
        brier = calculate_brier_score_single(pred_probs, times_np, events_np, t)
        auc = calculate_time_specific_auc(pred_probs, times_np, events_np, t)
        accuracy = calculate_time_accuracy(pred_probs, times_np, events_np, t)

        base_metrics[f'{year}年'] = {
            'c_index': c_index,
            'brier_score': brier,
            'auc': auc,
            'accuracy': accuracy
        }

    return base_metrics


def merge_bootstrap_results(base_metrics, bootstrap_results):

    merged_results = {}

    if bootstrap_results is None:
        print("[WARNING] bootstrap_results为None，返回基础指标")
        return base_metrics

    for time_key in ['1年', '3年', '5年']:
        if time_key in base_metrics:
            merged = base_metrics[time_key].copy()

            # 如果bootstrap_results中有对应的时间点
            if time_key in bootstrap_results:
                bootstrap_metrics = bootstrap_results[time_key]


                for metric in ['c_index', 'brier_score', 'auc', 'accuracy']:
                    if metric in bootstrap_metrics and bootstrap_metrics[metric] is not None:
                        merged[metric] = bootstrap_metrics[metric]


            if 'hr' in base_metrics[time_key] and base_metrics[time_key]['hr'] is not None:
                merged['hr'] = base_metrics[time_key]['hr']
            if 'logrank_p' in base_metrics[time_key] and base_metrics[time_key]['logrank_p'] is not None:
                merged['logrank_p'] = base_metrics[time_key]['logrank_p']

            merged_results[time_key] = merged


    if '综合指标' in base_metrics:
        merged_results['综合指标'] = base_metrics['综合指标']

    return merged_results
def calculate_time_specific_auc(pred_probs, times, events, eval_time):
    """计算时间特定的AUC"""
    from sklearn.metrics import roc_auc_score

    # 创建二分类标签：在eval_time之前发生事件=1，否则=0
    y_true = np.zeros_like(events, dtype=int)

    # 确定可评估的样本（事件发生或删失时间≥eval_time）
    evaluable_mask = (events == 1) | (times >= eval_time)

    if np.sum(evaluable_mask) < 10:
        return 0.5  # 样本太少，返回随机值

    # 事件发生在eval_time之前
    event_mask = (events == 1) & (times <= eval_time)
    y_true[event_mask] = 1

    try:
        auc = roc_auc_score(
            y_true[evaluable_mask],
            pred_probs[evaluable_mask]
        )
    except:
        auc = 0.5  # 如果计算失败，返回随机值

    return auc


def calculate_time_accuracy(pred_probs, times, events, eval_time):

    y_true = np.zeros_like(events, dtype=int)
    evaluable_mask = (events == 1) | (times >= eval_time)

    if np.sum(evaluable_mask) == 0:
        return 0.5


    event_mask = (events == 1) & (times <= eval_time)
    y_true[event_mask] = 1


    median_prob = np.median(pred_probs[evaluable_mask])
    y_pred = (pred_probs > median_prob).astype(int)


    accuracy = np.mean(y_true[evaluable_mask] == y_pred[evaluable_mask])

    return accuracy


def calculate_logrank_p_value(pred_probs, times, events, eval_time):

    from lifelines.statistics import logrank_test


    median_risk = np.median(pred_probs)
    high_risk = pred_probs > median_risk


    at_risk_mask = times >= eval_time


    n_low = np.sum(at_risk_mask & (~high_risk))
    n_high = np.sum(at_risk_mask & high_risk)

    if n_low < 5 or n_high < 5:
        return None

    try:

        T1 = times[at_risk_mask & (~high_risk)]
        T2 = times[at_risk_mask & high_risk]
        E1 = events[at_risk_mask & (~high_risk)]
        E2 = events[at_risk_mask & high_risk]


        results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
        return results.p_value
    except:
        return None


def calculate_time_dependent_cindex_single(pred_probs, times, events, eval_time):

    from lifelines.utils import concordance_index


    mask = (events == 1) | (times >= eval_time)

    if np.sum(mask) < 10:
        return 0.5


    risk_scores = -pred_probs[mask]

    try:
        c_index = concordance_index(
            times[mask],
            risk_scores,
            events[mask]
        )
    except:
        c_index = 0.5

    return c_index


def calculate_brier_score_single(pred_probs, times, events, eval_time):

    y_true = np.zeros_like(events, dtype=float)
    evaluable_mask = (events == 1) | (times >= eval_time)

    if np.sum(evaluable_mask) == 0:
        return 0.25


    event_mask = (events == 1) & (times <= eval_time)
    y_true[event_mask] = 1

    try:
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(
            y_true[evaluable_mask],
            pred_probs[evaluable_mask]
        )
    except:
        brier = 0.25

    return brier


def calculate_hazard_ratio_single(pred_probs, times, events, eval_time):

    from lifelines import CoxPHFitter
    import pandas as pd
    import warnings


    warnings.filterwarnings('ignore')


    median_risk = np.median(pred_probs)
    high_risk = pred_probs > median_risk


    at_risk_mask = times >= eval_time


    n_high_risk = np.sum(at_risk_mask & high_risk)
    n_low_risk = np.sum(at_risk_mask & (~high_risk))

    if n_high_risk < 5 or n_low_risk < 5:
        print(f"[INFO] 样本太少，无法计算HR: 高风险组={n_high_risk}, 低风险组={n_low_risk}")
        return None, None, None

    try:

        df = pd.DataFrame({
            'T': times[at_risk_mask],
            'E': events[at_risk_mask].astype(int),
            'risk': high_risk[at_risk_mask].astype(int)
        })


        if df['E'].sum() < 2:
            print(f"[INFO] 事件数量不足: {df['E'].sum()} < 2")
            return None, None, None

        if df['risk'].nunique() < 2:
            print(f"[INFO] 风险分组无效: 只有{df['risk'].nunique()}个组")
            return None, None, None


        cph = CoxPHFitter()


        cph.fit(df, duration_col='T', event_col='E',
                show_progress=False,
                step_size=0.1,
                penalizer=0.01)


        if cph.params_.empty:
            print("[WARNING] Cox模型未能估计任何参数")
            return None, None, None


        if 'risk' in cph.params_:
            hr = np.exp(cph.params_['risk'])
            hr_ci = np.exp(cph.confidence_intervals_.loc['risk'].values)
            print(f"[SUCCESS] HR计算成功: {hr:.3f} ({hr_ci[0]:.3f}-{hr_ci[1]:.3f})")
            return float(hr), float(hr_ci[0]), float(hr_ci[1])
        else:

            available_params = list(cph.params_.index)
            print(f"[WARNING] 'risk'参数不存在，可用参数: {available_params}")


            if available_params:
                param_name = available_params[0]
                hr = np.exp(cph.params_[param_name])
                hr_ci = np.exp(cph.confidence_intervals_.loc[param_name].values)
                print(f"[INFO] 使用参数 '{param_name}': HR={hr:.3f}")
                return float(hr), float(hr_ci[0]), float(hr_ci[1])
            else:
                return None, None, None

    except Exception as e:
        print(f"[ERROR] 计算HR失败: {str(e)[:100]}")
        return None, None, None
    finally:

        warnings.filterwarnings('default')

def calculate_overall_metrics(predictions, times, events):

    from lifelines.utils import concordance_index


    avg_risk = predictions.mean(axis=1)

    try:
        overall_c_index = concordance_index(
            times,
            -avg_risk,
            events
        )
    except:
        overall_c_index = 0.5


    brier_scores = []
    time_points = [365, 1095, 1825]

    for i, t in enumerate(time_points):
        pred_probs = 1 / (1 + np.exp(-predictions[:, i]))
        brier = calculate_brier_score_single(pred_probs, times, events, t)
        brier_scores.append(brier)

    avg_brier = np.mean(brier_scores) if brier_scores else 0.25


    time_auc_scores = []
    for i, t in enumerate(time_points):
        pred_probs = 1 / (1 + np.exp(-predictions[:, i]))
        auc = calculate_time_specific_auc(pred_probs, times, events, t)
        time_auc_scores.append(auc)

    avg_time_auc = np.mean(time_auc_scores) if time_auc_scores else 0.5

    return {
        '综合C指数': float(overall_c_index),
        '综合Brier评分': float(avg_brier),
        '时间AUC': float(avg_time_auc),
        '校准指数': float(1.0 - avg_brier / 0.25)
    }

class MultiTimeSurvivalLoss(nn.Module):


    def __init__(self, time_points=[365, 1095, 1825], alpha=0.5):
        super().__init__()
        self.time_points = time_points
        self.alpha = alpha  # 平衡Cox损失和逐点损失

    def forward(self, predictions, times, events):

        total_loss = 0


        cox_loss = self.cox_loss(predictions, times, events)


        pointwise_loss = self.pointwise_binary_loss(predictions, times, events)


        total_loss = self.alpha * cox_loss + (1 - self.alpha) * pointwise_loss

        return total_loss

    def cox_loss(self, risks, times, events):

        avg_risk = risks.mean(dim=1)  # [batch_size]

        # 计算Cox损失
        current_batch_len = len(times)
        R_mat = torch.zeros([current_batch_len, current_batch_len],
                            dtype=int, device=risks.device)

        for i in range(current_batch_len):
            for j in range(current_batch_len):
                if times[j] >= times[i]:
                    R_mat[i, j] = 1

        theta = avg_risk.reshape(-1)
        exp_theta = torch.exp(theta)

        loss_cox = -torch.mean(
            (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * events
        )

        return loss_cox

    def pointwise_binary_loss(self, predictions, times, events):

        total_point_loss = 0
        n_time_points = len(self.time_points)

        for i, t in enumerate(self.time_points):

            y_true = torch.zeros_like(events, dtype=torch.float32)


            evaluable_mask = (events == 1) | (times >= t)


            event_mask = (events == 1) & (times <= t)
            y_true[event_mask] = 1


            if torch.sum(evaluable_mask) > 0:
                time_loss = F.binary_cross_entropy_with_logits(
                    predictions[evaluable_mask, i],
                    y_true[evaluable_mask],
                    reduction='mean'
                )
                total_point_loss += time_loss

        return total_point_loss / n_time_points


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


    plt.xlabel('生存时间 (天)', fontsize=12)
    plt.ylabel('生存概率', fontsize=12)
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')


    plt.tight_layout()


    if keep_latest_only:
        km_path = os.path.join(save_dir, f'{prefix}_km_curve_latest.png')
        km_json_path = os.path.join(save_dir, f'{prefix}_km_data_latest.json')
    else:
        km_path = os.path.join(save_dir, f'{prefix}_km_curve_epoch_{epoch:03d}.png')
        km_json_path = os.path.join(save_dir, f'{prefix}_km_data_epoch_{epoch:03d}.json')

    plt.savefig(km_path, dpi=150, bbox_inches='tight')
    plt.close()


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

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import json

    os.makedirs(save_dir, exist_ok=True)


    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150


    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:

        print(f"[INFO] 单时间点输出，无法生成3个时间点的ROC曲线")
        return _save_single_roc(y_true, y_pred, epoch, save_dir, prefix, keep_latest_only)

    elif y_pred.shape[1] >= 3:

        print(f"[INFO] 生成3个时间点的ROC曲线: 1年、3年、5年")

        time_points = [0, 1, 2]  # 对应1年、3年、5年
        time_labels = ['1年', '3年', '5年']

        all_aucs = {}

        for i, (time_idx, time_label) in enumerate(zip(time_points, time_labels)):
            try:

                y_pred_proba = 1 / (1 + np.exp(-y_pred[:, time_idx]))


                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                all_aucs[time_label] = roc_auc


                fig, ax = plt.subplots(figsize=(10, 8))


                ax.plot(fpr, tpr, color='darkorange', lw=2.5,
                        label=f'ROC曲线 (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)


                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])


                ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=14)
                ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=14)
                ax.set_title(f'ROC曲线 - {time_label}生存预测 (Epoch {epoch})', fontsize=16, fontweight='bold')


                ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)


                ax.grid(True, alpha=0.3, linestyle='--')


                ax.text(0.02, 0.95, f'AUC = {roc_auc:.3f}', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


                plt.tight_layout(pad=2.0)


                if keep_latest_only:
                    roc_path = os.path.join(save_dir, f'{prefix}_roc_{time_label}_latest.png')
                else:
                    roc_path = os.path.join(save_dir, f'{prefix}_roc_{time_label}_epoch_{epoch:03d}.png')

                plt.savefig(roc_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

                print(f"  ✓ 保存{time_label} ROC曲线: {roc_path} (AUC: {roc_auc:.3f})")


                if keep_latest_only:
                    roc_json_path = os.path.join(save_dir, f'{prefix}_roc_{time_label}_latest.json')
                else:
                    roc_json_path = os.path.join(save_dir, f'{prefix}_roc_{time_label}_epoch_{epoch:03d}.json')

                roc_data = {
                    'epoch': epoch,
                    'time_label': time_label,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': float(roc_auc),
                    'n_samples': len(y_true),
                    'positive_rate': float(np.sum(y_true) / len(y_true))
                }

                with open(roc_json_path, 'w') as f:
                    json.dump(roc_data, f, indent=2)

            except Exception as e:
                print(f"[ERROR] 生成{time_label} ROC曲线失败: {e}")
                all_aucs[time_label] = 0.5


        _save_combined_roc(y_true, y_pred, epoch, save_dir, prefix, keep_latest_only, all_aucs)

        return all_aucs

    else:
        print(f"[WARNING] 预测维度不支持多时间点ROC: {y_pred.shape}")
        return {'平均AUC': 0.5}


def _save_combined_roc(y_true, y_pred, epoch, save_dir, prefix, keep_latest_only, all_aucs):

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    time_points = [0, 1, 2]
    time_labels = ['1年', '3年', '5年']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']


    fig, ax = plt.subplots(figsize=(12, 10))

    for i, (time_idx, time_label, color) in enumerate(zip(time_points, time_labels, colors)):
        try:
            y_pred_proba = 1 / (1 + np.exp(-y_pred[:, time_idx]))
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_value = all_aucs.get(time_label, 0.5)


            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f'{time_label} (AUC = {auc_value:.3f})')
        except Exception as e:
            print(f"[ERROR] 绘制{time_label}组合ROC失败: {e}")


    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='随机猜测')


    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=14)
    ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=14)
    ax.set_title(f'多时间点ROC曲线对比 - Epoch {epoch}', fontsize=16, fontweight='bold')


    ax.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)


    ax.grid(True, alpha=0.3, linestyle='--')


    auc_summary = '\n'.join([f'{label}: AUC = {all_aucs.get(label, 0.5):.3f}'
                             for label in time_labels])
    ax.text(0.6, 0.25, auc_summary, transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


    plt.tight_layout(pad=2.0)


    if keep_latest_only:
        combined_path = os.path.join(save_dir, f'{prefix}_roc_combined_latest.png')
    else:
        combined_path = os.path.join(save_dir, f'{prefix}_roc_combined_epoch_{epoch:03d}.png')

    plt.savefig(combined_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"  ✓ 保存组合ROC曲线: {combined_path}")

def _save_single_roc(y_true, y_pred, epoch, save_dir, prefix, keep_latest_only):


    from sklearn.metrics import roc_curve, auc
    import json


    if len(y_pred.shape) == 1:
        y_pred_proba = y_pred
    else:
        y_pred_proba = y_pred[:, 0]


    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)


    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    plt.title(f'ROC曲线 - 生存预测 (Epoch {epoch})', fontsize=14)
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





def save_confusion_matrix(y_true, y_pred, epoch, save_dir, prefix='val',
                          normalize=True, keep_latest_only=True):




    os.makedirs(save_dir, exist_ok=True)


    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['figure.dpi'] = 150

    y_true = np.array(y_true).flatten()


    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:

        return _save_single_cm(y_true, y_pred, epoch, save_dir, prefix, normalize, keep_latest_only)

    elif y_pred.shape[1] >= 3:

        print(f"[INFO] 生成3个时间点的混淆矩阵: 1年、3年、5年")

        time_points = [0, 1, 2]
        time_labels = ['1年', '3年', '5年']

        all_accuracies = {}
        all_cm_data = {}

        for i, (time_idx, time_label) in enumerate(zip(time_points, time_labels)):
            try:

                y_pred_proba = 1 / (1 + np.exp(-y_pred[:, time_idx]))


                accuracy, cm, cm_data = _save_time_specific_cm(
                    y_true, y_pred_proba, epoch, save_dir, prefix,
                    time_label, normalize, keep_latest_only
                )

                all_accuracies[time_label] = accuracy
                all_cm_data[time_label] = cm_data

                print(f"  ✓ 保存{time_label}混淆矩阵，准确率: {accuracy:.4f}")

            except Exception as e:
                print(f"[ERROR] 生成{time_label}混淆矩阵失败: {e}")
                all_accuracies[time_label] = 0.0
                all_cm_data[time_label] = None


        _save_combined_cm(all_accuracies, all_cm_data, epoch, save_dir, prefix, keep_latest_only)

        return all_accuracies, all_cm_data

    else:
        print(f"[WARNING] 预测维度不支持多时间点混淆矩阵: {y_pred.shape}")
        return _save_single_cm(y_true, y_pred, epoch, save_dir, prefix, normalize, keep_latest_only)


def _save_single_cm(y_true, y_pred, epoch, save_dir, prefix, normalize, keep_latest_only):

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import json


    if len(y_pred.shape) == 1:
        y_pred_proba = y_pred
    else:
        y_pred_proba = y_pred[:, 0] if y_pred.shape[1] > 0 else y_pred.flatten()


    threshold = np.median(y_pred_proba)
    y_pred_labels = (y_pred_proba > threshold).astype(int)


    cm = confusion_matrix(y_true, y_pred_labels)
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'


    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)


    classes = np.unique(np.concatenate([y_true, y_pred_labels]))


    ax = sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                     cbar=True, square=True,
                     xticklabels=classes, yticklabels=classes)


    plt.xlabel('预测标签 (Predicted Label)', fontsize=14)
    plt.ylabel('真实标签 (True Label)', fontsize=14)
    plt.title(f'混淆矩阵 - Epoch {epoch} ({prefix})\n准确率: {accuracy:.3f}', fontsize=16)


    if keep_latest_only:
        cm_path = os.path.join(save_dir, f'{prefix}_cm_latest.png')
        cm_json_path = os.path.join(save_dir, f'{prefix}_cm_latest.json')
    else:
        cm_path = os.path.join(save_dir, f'{prefix}_cm_epoch_{epoch:03d}.png')
        cm_json_path = os.path.join(save_dir, f'{prefix}_cm_epoch_{epoch:03d}.json')

    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()


    cm_data = {
        'epoch': epoch,
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'normalized': normalize,
        'predicted_labels': y_pred_labels.tolist(),
        'true_labels': y_true.tolist(),
        'threshold': float(threshold)
    }

    with open(cm_json_path, 'w') as f:
        json.dump(cm_data, f, indent=2)

    return accuracy, cm_data


def _save_time_specific_cm(y_true, y_pred_proba, epoch, save_dir, prefix,
                           time_label, normalize, keep_latest_only):

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import json


    threshold = np.median(y_pred_proba)
    y_pred_labels = (y_pred_proba > threshold).astype(int)


    cm = confusion_matrix(y_true, y_pred_labels)
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm_normalized = np.nan_to_num(cm_normalized)
        cm_display = cm_normalized
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = 'd'
        vmin, vmax = None, None


    fig, ax = plt.subplots(figsize=(10, 8))


    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                cbar=True, square=True, ax=ax,
                xticklabels=['低风险', '高风险'],
                yticklabels=['低风险', '高风险'],
                vmin=vmin, vmax=vmax,
                linewidths=1, linecolor='white')


    ax.set_xlabel('预测标签 (Predicted Label)', fontsize=14, fontweight='bold')
    ax.set_ylabel('真实标签 (True Label)', fontsize=14, fontweight='bold')
    ax.set_title(f'混淆矩阵 - {time_label}生存预测 (Epoch {epoch})',
                 fontsize=16, fontweight='bold', pad=20)


    accuracy_text = f'准确率: {accuracy:.3f}\n阈值: {threshold:.3f}'
    ax.text(0.02, -0.15, accuracy_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


    plt.tight_layout(pad=2.0)


    if keep_latest_only:
        cm_path = os.path.join(save_dir, f'{prefix}_cm_{time_label}_latest.png')
        cm_json_path = os.path.join(save_dir, f'{prefix}_cm_{time_label}_latest.json')
    else:
        cm_path = os.path.join(save_dir, f'{prefix}_cm_{time_label}_epoch_{epoch:03d}.png')
        cm_json_path = os.path.join(save_dir, f'{prefix}_cm_{time_label}_epoch_{epoch:03d}.json')

    plt.savefig(cm_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


    cm_data = {
        'epoch': epoch,
        'time_label': time_label,
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'normalized': normalize,
        'predicted_labels': y_pred_labels.tolist(),
        'true_labels': y_true.tolist(),
        'threshold': float(threshold),
        'prob_mean': float(np.mean(y_pred_proba)),
        'prob_std': float(np.std(y_pred_proba))
    }

    with open(cm_json_path, 'w') as f:
        json.dump(cm_data, f, indent=2)

    return accuracy, cm, cm_data


def _save_combined_cm(all_accuracies, all_cm_data, epoch, save_dir, prefix, keep_latest_only):

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    time_labels = ['1年', '3年', '5年']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']


    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'多时间点混淆矩阵对比 - Epoch {epoch}', fontsize=18, fontweight='bold', y=1.05)

    for i, (time_label, color, ax) in enumerate(zip(time_labels, colors, axes)):
        if time_label in all_cm_data and all_cm_data[time_label] is not None:
            cm_data = all_cm_data[time_label]

            if 'confusion_matrix' in cm_data:
                cm = np.array(cm_data['confusion_matrix'])
                accuracy = all_accuracies.get(time_label, 0.0)


                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)


                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                            cbar=True, square=True, ax=ax,
                            xticklabels=['低风险', '高风险'],
                            yticklabels=['低风险', '高风险'],
                            vmin=0, vmax=1,
                            linewidths=1, linecolor='white')

                ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
                ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
                ax.set_title(f'{time_label}\n准确率: {accuracy:.3f}',
                             fontsize=14, fontweight='bold')


    plt.tight_layout(pad=2.0)


    if keep_latest_only:
        combined_path = os.path.join(save_dir, f'{prefix}_cm_combined_latest.png')
    else:
        combined_path = os.path.join(save_dir, f'{prefix}_cm_combined_epoch_{epoch:03d}.png')

    plt.savefig(combined_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"  ✓ 保存组合混淆矩阵: {combined_path}")

def CIndex(hazards, labels, survtime_all):

    if hazards.dim() == 2 and hazards.shape[1] > 1:

        hazards_flat = hazards.mean(dim=1)
    else:
        hazards_flat = hazards.reshape(-1)

    labels = labels.data.cpu().numpy()
    survtime_all = survtime_all.data.cpu().numpy()
    hazards_flat = hazards_flat.data.cpu().numpy()


    if len(labels) != len(hazards_flat):
        print(f"[WARNING] CIndex维度不匹配: labels={len(labels)}, hazards={len(hazards_flat)}")
        return 0.5

    concord = 0.
    total = 0.
    N_test = labels.shape[0]

    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards_flat[j] < hazards_flat[i]:
                        concord = concord + 1
                    elif hazards_flat[j] == hazards_flat[i]:
                        concord = concord + 0.5

    return (concord / total) if total > 0 else 0.5


def CIndex_lifeline(hazards, labels, survtime_all):

    if hazards.dim() == 2 and hazards.shape[1] > 1:

        hazards_flat = hazards.mean(dim=1).cpu().numpy().reshape(-1)
    else:
        hazards_flat = hazards.cpu().numpy().reshape(-1)


    if labels.dim() > 1:
        labels = labels.squeeze()
    labels = labels.data.cpu().numpy().reshape(-1)


    if survtime_all.dim() > 1:
        survtime_all = survtime_all.squeeze()
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)


    valid_mask = ~np.isnan(hazards_flat)

    if np.sum(valid_mask) < 10:
        print(f"[WARNING] 有效样本太少: {np.sum(valid_mask)}")
        return 0.5

    new_hazard = hazards_flat[valid_mask]
    new_label = labels[valid_mask]
    new_surv = survtime_all[valid_mask]

    if len(new_hazard) == 0:
        return 0.5

    try:
        from lifelines.utils import concordance_index
        return concordance_index(new_surv, -new_hazard, new_label)
    except Exception as e:
        print(f"[WARNING] 计算C-index失败: {e}")
        return 0.5


def accuracy_cox(hazards, labels):

    if hazards.dim() == 2 and hazards.shape[1] > 1:

        hazardsdata = hazards.mean(dim=1).cpu().numpy().reshape(-1)
    else:

        hazardsdata = hazards.cpu().numpy().reshape(-1)


    if labels.dim() > 1:
        labels = labels.squeeze()

    labels = labels.data.cpu().numpy().reshape(-1)


    if len(hazardsdata) != len(labels):
        print(f"[WARNING] 维度不匹配: hazards={len(hazardsdata)}, labels={len(labels)}")

        min_len = min(len(hazardsdata), len(labels))
        hazardsdata = hazardsdata[:min_len]
        labels = labels[:min_len]


    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1

    # 计算准确率
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazards, labels, survtime_all):

    if hazards.dim() == 2 and hazards.shape[1] > 1:

        hazardsdata = hazards.mean(dim=1).cpu().numpy().reshape(-1)
    else:
        hazardsdata = hazards.cpu().numpy().reshape(-1)


    if labels.dim() > 1:
        labels = labels.squeeze()
    labels = labels.data.cpu().numpy().reshape(-1)

    if survtime_all.dim() > 1:
        survtime_all = survtime_all.squeeze()
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)


    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1


    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]


    if len(T1) < 5 or len(T2) < 5:
        print(f"[WARNING] 样本太少: 组1={len(T1)}, 组2={len(T2)}")
        return 1.0

    try:
        from lifelines.statistics import logrank_test
        results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
        return results.p_value
    except Exception as e:
        print(f"[WARNING] 计算log-rank检验失败: {e}")
        return 1.0


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

    if loss == 'multi-time':

        multi_time_loss = MultiTimeSurvivalLoss()
        return multi_time_loss(hazard_pred, survtime, censor)
    elif loss == 'deepsurv':
        nll_loss = NegativeLogLikelihood(l2_reg)
        return nll_loss(hazard_pred, survtime, censor, model)
    elif loss == 'cox-nnet':
        current_batch_len = len(survtime)


        if len(hazard_pred.shape) == 2 and hazard_pred.shape[1] > 1:

            avg_risk = hazard_pred.mean(dim=1)
        else:

            avg_risk = hazard_pred.reshape(-1)


        R_mat = torch.zeros([current_batch_len, current_batch_len],
                            dtype=torch.float32, device=avg_risk.device)

        for i in range(current_batch_len):
            for j in range(current_batch_len):
                if survtime[j] >= survtime[i]:
                    R_mat[i, j] = 1

        theta = avg_risk
        exp_theta = torch.exp(theta)


        exp_theta = exp_theta.view(-1, 1)
        R_mat = R_mat


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
                    clinical_dim=0, hypoxia_dim=0, use_context_tokens=True,
                    save_metrics_dir=None,is_best_epoch=False,bootstrap_n: int = 100):


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
                    # 模型返回4个值
                    outputs = model(wsi_features.cuda(), gene_features.cuda(), clinical_data, hypoxia_data)

                    # 确保正确处理所有返回值
                    if len(outputs) == 4:
                        gene2wsi_feature, pred, fusion_attn, wsi_attn_weights = outputs
                    elif len(outputs) == 5:
                        # 如果有5个返回值（包含了额外的信息）
                        gene2wsi_feature, pred, fusion_attn, wsi_attn_weights, extra = outputs[:4]
                    else:
                        # 适应不同的返回值长度
                        gene2wsi_feature = outputs[0] if len(outputs) > 0 else None
                        pred = outputs[1] if len(outputs) > 1 else None
                        fusion_attn = outputs[2] if len(outputs) > 2 else None
                        wsi_attn_weights = outputs[3] if len(outputs) > 3 else None
                else:
                    outputs = model(wsi_features.cuda(), gene_features.cuda())
                    # 同样的处理逻辑
                    if len(outputs) == 4:
                        gene2wsi_feature, pred, fusion_attn, wsi_attn_weights = outputs
                    else:
                        gene2wsi_feature = outputs[0] if len(outputs) > 0 else None
                        pred = outputs[1] if len(outputs) > 1 else None
                        fusion_attn = outputs[2] if len(outputs) > 2 else None
                        wsi_attn_weights = outputs[3] if len(outputs) > 3 else None

                # =============== 在这里添加注意力保存代码 ===============
                # 如果是最佳epoch的第一个批次，保存注意力权重
                if is_best_epoch and step == 0:
                    try:
                        # 确保wsi_attn_weights存在
                        if wsi_attn_weights is not None:
                            save_attention_data = {
                                'epoch': epoch,
                                'attention_weights': wsi_attn_weights,
                                'batch_size': wsi_features.shape[0]
                            }

                            # 保存到文件
                            import pickle
                            save_dir = 'best_epoch_attention'
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f'epoch_{epoch}_attention.pkl')

                            # 处理张量，转换为CPU
                            def to_cpu(obj):
                                if isinstance(obj, torch.Tensor):
                                    return obj.cpu().detach()
                                elif isinstance(obj, list):
                                    return [to_cpu(item) for item in obj]
                                elif isinstance(obj, dict):
                                    return {k: to_cpu(v) for k, v in obj.items()}
                                else:
                                    return obj

                            save_data_cpu = to_cpu(save_attention_data)

                            with open(save_path, 'wb') as f:
                                pickle.dump(save_data_cpu, f)

                            print(f"✓ 保存最佳epoch注意力权重: {save_path}")

                            # 同时保存为npy格式便于可视化
                            if wsi_attn_weights and len(wsi_attn_weights) > 0:
                                # 取第一层的注意力矩阵
                                first_layer_attn = wsi_attn_weights[0]
                                if first_layer_attn is not None:
                                    # 处理DataParallel的情况
                                    if isinstance(first_layer_attn, list):
                                        # 多GPU：取第一个GPU的数据
                                        first_layer_attn = first_layer_attn[0]

                                    if isinstance(first_layer_attn, torch.Tensor):
                                        attn_np = first_layer_attn.cpu().detach().numpy()
                                        npy_path = os.path.join(save_dir, f'epoch_{epoch}_attn_matrix.npy')
                                        np.save(npy_path, attn_np)
                                        print(f"✓ 保存注意力矩阵: {npy_path}, 形状: {attn_np.shape}")
                    except Exception as e:
                        print(f"保存注意力权重失败: {e}")
                # =============== 添加结束 ===============

                # 计算对比损失
                if gene2wsi_feature is not None:
                    gene2wsi_feature = gene2wsi_feature.transpose(-2, -1)
                    sorted_gen2wsi_feat, _ = torch.sort(gene2wsi_feature, descending=True, dim=1)
                    gene2wsiloss = criterion(sorted_gen2wsi_feat, labels)
                else:
                    gene2wsiloss = 0

            else:  # 没有对比损失的情况
                if clinical_data is not None or hypoxia_data is not None:
                    outputs = model(wsi_features.cuda(), gene_features.cuda(), clinical_data, hypoxia_data)
                    if len(outputs) >= 3:
                        pred = outputs[1]  # 假设第二个是预测值
                        # 获取注意力权重（如果存在）
                        if len(outputs) >= 4:
                            wsi_attn_weights = outputs[3]
                        else:
                            wsi_attn_weights = None
                    else:
                        pred = outputs[0]
                        wsi_attn_weights = None
                else:
                    outputs = model(wsi_features.cuda(), gene_features.cuda())
                    if len(outputs) >= 3:
                        pred = outputs[1]
                        if len(outputs) >= 4:
                            wsi_attn_weights = outputs[3]
                        else:
                            wsi_attn_weights = None
                    else:
                        pred = outputs[0]
                        wsi_attn_weights = None
                gene2wsiloss = 0

                # =============== 这里也需要添加相同的代码 ===============
                # 如果是最佳epoch的第一个批次，保存注意力权重
                if is_best_epoch and step == 0:
                    try:
                        if wsi_attn_weights is not None:
                            save_attention_data = {
                                'epoch': epoch,
                                'attention_weights': wsi_attn_weights,
                                'batch_size': wsi_features.shape[0]
                            }

                            import pickle
                            save_dir = 'best_epoch_attention'
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f'epoch_{epoch}_attention_no_contrast.pkl')

                            def to_cpu(obj):
                                if isinstance(obj, torch.Tensor):
                                    return obj.cpu().detach()
                                elif isinstance(obj, list):
                                    return [to_cpu(item) for item in obj]
                                elif isinstance(obj, dict):
                                    return {k: to_cpu(v) for k, v in obj.items()}
                                else:
                                    return obj

                            save_data_cpu = to_cpu(save_attention_data)

                            with open(save_path, 'wb') as f:
                                pickle.dump(save_data_cpu, f)

                            print(f"✓ 保存最佳epoch注意力权重(无对比损失): {save_path}")

                            if wsi_attn_weights and len(wsi_attn_weights) > 0:
                                first_layer_attn = wsi_attn_weights[0]
                                if first_layer_attn is not None:
                                    if isinstance(first_layer_attn, list):
                                        first_layer_attn = first_layer_attn[0]

                                    if isinstance(first_layer_attn, torch.Tensor):
                                        attn_np = first_layer_attn.cpu().detach().numpy()
                                        npy_path = os.path.join(save_dir, f'epoch_{epoch}_attn_matrix_no_contrast.npy')
                                        np.save(npy_path, attn_np)
                                        print(f"✓ 保存注意力矩阵: {npy_path}")
                    except Exception as e:
                        print(f"保存注意力权重失败: {e}")
                # =============== 添加结束 ===============

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

        if pred is not None and len(pred.shape) == 2 and pred.shape[1] > 1:
            # 多时间点输出，使用multi-time损失
            cox_loss_type = 'multi-time'
        else:
            # 单时间点输出
            cox_loss_type = 'cox-nnet'

        # 计算损失
        if reg_loss and (train_flag == 0):
            loss = CoxLoss(futime, fustat, pred, loss=cox_loss_type) + gene2wsiloss + reg_loss(model)
        elif reg_loss and (train_flag == 1):
            loss = CoxLoss(futime, fustat, pred, loss=cox_loss_type) + reg_loss(model)
        elif not reg_loss and (train_flag == 0):
            loss = CoxLoss(futime, fustat, pred, loss=cox_loss_type) + gene2wsiloss
        elif not reg_loss and (train_flag == 1):
            loss = CoxLoss(futime, fustat, pred, loss=cox_loss_type)


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

            # 在计算指标时使用Bootstrap
        # 在计算指标时使用Bootstrap
        if pred_all is not None and fustat_torch is not None and survtime_torch is not None:
            # 使用Bootstrap计算指标
            try:
                train_results = calculate_multi_time_metrics_all_in_one(
                    pred_all.data,
                    survtime_torch,
                    fustat_torch,
                    time_points=[365, 1095, 1825],
                    bootstrap_n=bootstrap_n,
                    random_seed=epoch * 100 + 42  # 每轮不同的随机种子
                )

                # 确保train_results不为None
                if train_results is None:
                    print("[WARNING] train_results为None，创建空字典")
                    train_results = {}

                # 打印带置信区间的结果
                print(f"\n=== 第{epoch}轮训练结果 ===")

                if 'bootstrap_summary' in train_results:
                    # 使用Bootstrap的结果
                    for time_point in ['1年', '3年', '5年']:
                        if time_point in train_results['metrics']:
                            metrics = train_results['metrics'][time_point]
                            print(f"\n{time_point}:")
                            for metric_name in ['c_index', 'brier_score', 'auc', 'accuracy']:
                                if metric_name in metrics and metrics[metric_name] is not None:
                                    value = metrics[metric_name]
                                    if isinstance(value, dict):
                                        print(f"  {metric_name}: {value['value']:.3f} "
                                              f"({value['ci_lower']:.3f}-{value['ci_upper']:.3f})")
                                    else:
                                        print(f"  {metric_name}: {value:.3f}")
                else:
                    # 普通结果
                    for time_point, metrics in train_results.items():
                        if time_point != '综合指标':
                            print(f"{time_point}: C-index={metrics.get('c_index', 0.5):.3f}, "
                                  f"Brier={metrics.get('brier_score', 0.25):.3f}")

            except Exception as e:
                print(f"[ERROR] 计算训练指标失败: {e}")
                train_results = {}

            return (accu_loss.item() / (step + 1), train_results,
                    pred_all.data.cpu().numpy(),
                    fustat_torch.cpu().numpy(),
                    survtime_torch.cpu().numpy())

@torch.no_grad()
def evaluate(model, topK, criterion, data_loader, epoch, json_path, reg_loss=False, train_flag=0,
             contrastive_loss_flag=0, clinical_dim=0, hypoxia_dim=0, use_context_tokens=True,
             save_metrics_dir=None,bootstrap_n: int = 100):


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


    if pred_all is not None and fustat_torch is not None and survtime_torch is not None:

        if pred_all.dim() == 2 and pred_all.shape[1] > 1:
            print(f"[INFO] 多时间点预测: {pred_all.shape}")

            pred_avg = pred_all.mean(dim=1, keepdim=True)
            acc = accuracy_cox(pred_avg.data, fustat_torch)
            pvalue_pred = cox_log_rank(pred_avg.data, fustat_torch, survtime_torch)
            c_index = CIndex_lifeline(pred_avg.data, fustat_torch, survtime_torch)
        else:
            acc = accuracy_cox(pred_all.data, fustat_torch)
            pvalue_pred = cox_log_rank(pred_all.data, fustat_torch, survtime_torch)
            c_index = CIndex_lifeline(pred_all.data, fustat_torch, survtime_torch)

        print(f"\nValidation Results - Loss: {accu_loss.item() / (step + 1):.4f}, "
              f"C-index: {c_index:.4f}, P-value: {pvalue_pred:.4f}, Cox Acc: {acc:.4f}")



        val_multi_metrics = None
        try:

            val_multi_metrics = calculate_multi_time_metrics_all_in_one(
                pred_all.data,
                survtime_torch,
                fustat_torch,
                time_points=[365, 1095, 1825],
                bootstrap_n=bootstrap_n,
                random_seed=epoch * 100 + 42
            )
        except Exception as e:
            print(f"[WARNING] 计算验证集多时间点指标失败: {e}")


        all_preds_np = pred_all.detach().cpu().numpy()
        all_labels_np = fustat_torch.cpu().numpy()


        if all_preds_np.shape[1] > 1:

            all_probs_np = 1 / (1 + np.exp(-all_preds_np[:, 0]))
        else:

            if len(all_preds_np.shape) == 1:
                all_probs_np = 1 / (1 + np.exp(-all_preds_np))
            else:
                all_probs_np = 1 / (1 + np.exp(-all_preds_np[:, 0]))


        try:

            if all_preds_np.shape[1] > 1:
                risk_scores = all_preds_np.mean(axis=1)
            else:
                risk_scores = all_preds_np.flatten()

            km_path, km_data = plot_kaplan_meier_survival_curve(
                all_labels_np,
                risk_scores,
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
            km_path = None
            km_data = None

        return (accu_loss.item() / (step + 1), acc, pvalue_pred, c_index,
                all_preds_np, all_labels_np, all_probs_np, val_multi_metrics, survtime_torch.cpu().numpy())
    else:
        print("Warning: No predictions generated during validation")
        return (0.0, 0.0, 1.0, 0.5, None, None, None, None)