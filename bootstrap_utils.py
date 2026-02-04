# bootstrap_utils.py
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
import warnings

warnings.filterwarnings('ignore')


class BootstrapEvaluator:
    """Bootstrap评估器，用于计算指标的置信区间"""

    def __init__(self, n_bootstrap: int = 1000, random_seed: int = 42,
                 confidence_level: float = 0.95):

        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_seed)

    def bootstrap_resample(self, data: np.ndarray, labels: np.ndarray = None):

        n_samples = len(data)
        indices = self.rng.randint(0, n_samples, size=n_samples)

        if labels is not None:
            return data[indices], labels[indices], indices
        return data[indices], indices

    def calculate_confidence_intervals(self, bootstrap_metrics: np.ndarray):
        """计算置信区间"""
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        ci_lower = np.percentile(bootstrap_metrics, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile, axis=0)
        mean_value = np.mean(bootstrap_metrics, axis=0)
        std_value = np.std(bootstrap_metrics, axis=0)

        return {
            'mean': mean_value,
            'std': std_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': self.confidence_level
        }

    def evaluate_with_bootstrap(self,
                                predictions: np.ndarray,
                                times: np.ndarray,
                                events: np.ndarray,
                                metric_function: Callable,
                                time_points: List[int] = [365, 1095, 1825],
                                verbose: bool = True):

        n_samples = len(times)
        n_time_points = len(time_points)


        bootstrap_results = {
            'c_index': np.zeros((self.n_bootstrap, n_time_points)),
            'brier_score': np.zeros((self.n_bootstrap, n_time_points)),
            'auc': np.zeros((self.n_bootstrap, n_time_points)),
            'accuracy': np.zeros((self.n_bootstrap, n_time_points))
        }


        original_metrics = metric_function(predictions, times, events, time_points)

        if verbose:
            print(f"执行 {self.n_bootstrap} 次Bootstrap重采样...")

        # 执行Bootstrap
        for b in tqdm(range(self.n_bootstrap), desc="Bootstrap", disable=not verbose):
            # 重采样
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            pred_boot = predictions[indices]
            times_boot = times[indices]
            events_boot = events[indices]

            # 计算指标
            try:
                boot_metrics = metric_function(pred_boot, times_boot, events_boot, time_points)

                # 存储结果
                for i, tp in enumerate(time_points):
                    year = tp // 365
                    key = f'{year}年'

                    if key in boot_metrics:
                        bootstrap_results['c_index'][b, i] = boot_metrics[key]['c_index']
                        bootstrap_results['brier_score'][b, i] = boot_metrics[key]['brier_score']
                        bootstrap_results['auc'][b, i] = boot_metrics[key]['auc']
                        bootstrap_results['accuracy'][b, i] = boot_metrics[key]['accuracy']
            except Exception as e:
                if verbose:
                    print(f"Bootstrap {b} 失败: {e}")
                continue

        # 计算置信区间
        confidence_intervals = {}
        for metric_name in bootstrap_results.keys():
            confidence_intervals[metric_name] = self.calculate_confidence_intervals(
                bootstrap_results[metric_name]
            )


        final_results = {}
        for i, tp in enumerate(time_points):
            year = tp // 365
            key = f'{year}年'

            final_results[key] = {
                'c_index': {
                    'value': original_metrics[key]['c_index'],
                    'ci_lower': confidence_intervals['c_index']['ci_lower'][i],
                    'ci_upper': confidence_intervals['c_index']['ci_upper'][i],
                    'std': confidence_intervals['c_index']['std'][i]
                },
                'brier_score': {
                    'value': original_metrics[key]['brier_score'],
                    'ci_lower': confidence_intervals['brier_score']['ci_lower'][i],
                    'ci_upper': confidence_intervals['brier_score']['ci_upper'][i],
                    'std': confidence_intervals['brier_score']['std'][i]
                },
                'auc': {
                    'value': original_metrics[key]['auc'],
                    'ci_lower': confidence_intervals['auc']['ci_lower'][i],
                    'ci_upper': confidence_intervals['auc']['ci_upper'][i],
                    'std': confidence_intervals['auc']['std'][i]
                },
                'accuracy': {
                    'value': original_metrics[key]['accuracy'],
                    'ci_lower': confidence_intervals['accuracy']['ci_lower'][i],
                    'ci_upper': confidence_intervals['accuracy']['ci_upper'][i],
                    'std': confidence_intervals['accuracy']['std'][i]
                }
            }


        final_results = self.calculate_hr_confidence_intervals(
            predictions, times, events, time_points, final_results
        )

        return final_results

    def calculate_hr_confidence_intervals(self, predictions, times, events,
                                          time_points, results_dict):

        from lifelines import CoxPHFitter
        import pandas as pd

        for i, tp in enumerate(time_points):
            year = tp // 365
            key = f'{year}年'


            hr_bootstrap = []
            pred_probs = 1 / (1 + np.exp(-predictions[:, i]))

            for b in range(min(200, self.n_bootstrap)):  # HR计算较慢，减少次数
                indices = self.rng.choice(len(times), size=len(times), replace=True)

                try:
                    pred_boot = pred_probs[indices]
                    times_boot = times[indices]
                    events_boot = events[indices]


                    median_risk = np.median(pred_boot)
                    risk_group = (pred_boot > median_risk).astype(int)


                    df = pd.DataFrame({
                        'T': times_boot,
                        'E': events_boot.astype(int),
                        'risk': risk_group
                    })


                    cph = CoxPHFitter()
                    cph.fit(df, duration_col='T', event_col='E')

                    hr = np.exp(cph.params_['risk'])
                    hr_bootstrap.append(hr)

                except:
                    continue

            if len(hr_bootstrap) > 10:

                hr_ci_lower = np.percentile(hr_bootstrap, 2.5)
                hr_ci_upper = np.percentile(hr_bootstrap, 97.5)
                hr_mean = np.mean(hr_bootstrap)
                hr_std = np.std(hr_bootstrap)

                results_dict[key]['hr'] = {
                    'value': results_dict[key].get('hr', {}).get('value', hr_mean),
                    'ci_lower': hr_ci_lower,
                    'ci_upper': hr_ci_upper,
                    'std': hr_std
                }

        return results_dict

    def bootstrap_summary_statistics(self, bootstrap_results):

        summary = {}

        for time_key, time_metrics in bootstrap_results.items():
            summary[time_key] = {}

            for metric_name, data in time_metrics.items():
                # 检查data是否为None
                if data is None:
                    print(f"[WARNING] {time_key}的{metric_name}为None，跳过")
                    summary[time_key][metric_name] = None
                    continue

                try:

                    if isinstance(data, dict) and 'value' in data:
                        summary[time_key][metric_name] = {
                            'estimate': float(data['value']),
                            'ci_lower': float(data['ci_lower']) if 'ci_lower' in data else None,
                            'ci_upper': float(data['ci_upper']) if 'ci_upper' in data else None,
                            'bootstrap_mean': float(np.mean([data['value']])),  # 这里需要实际的Bootstrap样本
                            'bootstrap_std': float(np.std([data['value']]))
                        }
                    elif isinstance(data, (int, float)):

                        summary[time_key][metric_name] = {
                            'estimate': float(data),
                            'ci_lower': None,
                            'ci_upper': None,
                            'bootstrap_mean': float(data),
                            'bootstrap_std': 0.0
                        }
                    else:
                        print(f"[WARNING] {time_key}的{metric_name}格式无效: {type(data)}")
                        summary[time_key][metric_name] = None
                except Exception as e:
                    print(f"[ERROR] 处理{time_key}的{metric_name}时出错: {e}")
                    summary[time_key][metric_name] = None

        return summary