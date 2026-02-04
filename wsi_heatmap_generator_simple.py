import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime


class SimpleHeatmapGenerator:


    def __init__(self, output_dir='heatmaps'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def tensor_to_cpu_numpy(self, tensor_or_list):

        if tensor_or_list is None:
            return None

        if isinstance(tensor_or_list, torch.Tensor):

            if tensor_or_list.is_cuda:
                return tensor_or_list.cpu().detach().numpy()
            else:
                return tensor_or_list.detach().numpy()

        elif isinstance(tensor_or_list, list):

            result = []
            for item in tensor_or_list:
                if isinstance(item, torch.Tensor):
                    if item.is_cuda:
                        result.append(item.cpu().detach().numpy())
                    else:
                        result.append(item.detach().numpy())
                else:
                    result.append(item)
            return result

        elif isinstance(tensor_or_list, np.ndarray):
            return tensor_or_list

        else:
            return tensor_or_list

    def process_attention_weights(self, attention_weights, batch_idx=0):
        """处理注意力权重"""
        if attention_weights is None:
            return None


        attn_cpu = self.tensor_to_cpu_numpy(attention_weights)

        if attn_cpu is None:
            return None


        processed_layers = []
        for layer_idx, layer_attn in enumerate(attn_cpu):
            if layer_attn is None:
                processed_layers.append(None)
                continue


            if len(layer_attn.shape) == 4:

                if batch_idx < layer_attn.shape[0]:
                    layer_batch = layer_attn[batch_idx]
                    if len(layer_batch.shape) == 3:
                        layer_mean = layer_batch.mean(axis=0)
                        processed_layers.append(layer_mean)
                    else:
                        processed_layers.append(layer_batch)
                else:

                    layer_batch = layer_attn[0]
                    if len(layer_batch.shape) == 3:
                        layer_mean = layer_batch.mean(axis=0)
                        processed_layers.append(layer_mean)
                    else:
                        processed_layers.append(layer_batch)

            elif len(layer_attn.shape) == 3:

                processed_layers.append(layer_attn.mean(axis=0))

            elif len(layer_attn.shape) == 2:

                processed_layers.append(layer_attn)

            else:
                processed_layers.append(None)

        return processed_layers

    def create_attention_heatmap(self, attention_matrix, title="Attention Heatmap"):
        """创建注意力热图"""
        if attention_matrix is None:
            return None

        plt.figure(figsize=(10, 8))


        if attention_matrix.shape[0] > 100:

            stride = attention_matrix.shape[0] // 100
            display_matrix = attention_matrix[::stride, ::stride]
        else:
            display_matrix = attention_matrix


        im = plt.imshow(display_matrix, cmap='hot', aspect='auto')
        plt.colorbar(im, label='Attention Score')
        plt.title(title)
        plt.xlabel('Patch Index')
        plt.ylabel('Patch Index')


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f'{title.replace(" ", "_")}_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def generate_attention_analysis(self, model_outputs, epoch, sample_idx=0, prefix='val'):

        print(f"\n{'=' * 60}")
        print(f"生成注意力分析 - Epoch {epoch}, 样本 {sample_idx}")
        print(f"{'=' * 60}")

        if len(model_outputs) < 4:
            print("模型输出不包含注意力权重")
            return

        attention_weights = model_outputs[3]


        processed_weights = self.process_attention_weights(attention_weights, sample_idx)

        if not processed_weights:
            print("没有可用的注意力权重")
            return


        heatmap_paths = []
        for layer_idx, attn_matrix in enumerate(processed_weights):
            if attn_matrix is not None:
                title = f"{prefix}_epoch{epoch}_sample{sample_idx}_layer{layer_idx}"
                heatmap_path = self.create_attention_heatmap(attn_matrix, title)
                if heatmap_path:
                    heatmap_paths.append(heatmap_path)
                    print(f"✓ 生成第{layer_idx}层热图: {heatmap_path}")

                    # 保存原始数据
                    data_path = os.path.join(self.output_dir, f"{title}.npy")
                    np.save(data_path, attn_matrix)
                    print(f"  保存原始数据: {data_path}, 形状: {attn_matrix.shape}")

        # 创建摘要报告
        if heatmap_paths:
            report = {
                'epoch': epoch,
                'sample_idx': sample_idx,
                'prefix': prefix,
                'timestamp': datetime.now().isoformat(),
                'num_layers': len(processed_weights),
                'heatmaps_generated': len(heatmap_paths),
                'heatmap_paths': heatmap_paths,
                'layer_shapes': [
                    attn_matrix.shape if attn_matrix is not None else None
                    for attn_matrix in processed_weights
                ]
            }

            # 保存报告
            report_path = os.path.join(self.output_dir, f"{prefix}_epoch{epoch}_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"\n✓ 注意力分析报告已保存: {report_path}")


            try:
                valid_matrices = [m for m in processed_weights if m is not None]
                if valid_matrices:
                    avg_matrix = np.mean(valid_matrices, axis=0)
                    avg_title = f"{prefix}_epoch{epoch}_sample{sample_idx}_average_all_layers"
                    avg_path = self.create_attention_heatmap(avg_matrix, avg_title)
                    print(f"✓ 生成综合热图: {avg_path}")
            except Exception as e:
                print(f"生成综合热图失败: {e}")

        return heatmap_paths

    def analyze_model_on_dataset(self, model, data_loader, epoch, num_samples=3, prefix='val'):
        """在数据集上分析模型注意力"""
        model.eval()

        all_results = []

        for batch_idx, data in enumerate(data_loader):
            if batch_idx >= 1:
                break

            wsi_features, gene_features, clinical_data, hypoxia_data, futime, fustat = data


            wsi_features = wsi_features.cuda()
            gene_features = gene_features.cuda()
            if clinical_data is not None:
                clinical_data = clinical_data.cuda()
            if hypoxia_data is not None:
                hypoxia_data = hypoxia_data.cuda()

            with torch.no_grad():

                if clinical_data is not None or hypoxia_data is not None:
                    outputs = model(wsi_features, gene_features, clinical_data, hypoxia_data)
                else:
                    outputs = model(wsi_features, gene_features)


            batch_size = min(wsi_features.shape[0], num_samples)
            for sample_idx in range(batch_size):
                print(f"\n分析样本 {sample_idx}...")


                heatmaps = self.generate_attention_analysis(
                    outputs, epoch, sample_idx, f"{prefix}_batch{batch_idx}"
                )

                all_results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': sample_idx,
                    'heatmaps': heatmaps if heatmaps else []
                })

        return all_results