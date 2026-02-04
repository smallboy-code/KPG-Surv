# wsi_heatmap_generator_simple.py
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

    def create_attention_heatmap(self, attention_matrix, title="Attention Heatmap", save_path=None):

        if attention_matrix is None:
            return None

        plt.figure(figsize=(10, 8))


        if attention_matrix.shape[0] > 100:
            stride = max(1, attention_matrix.shape[0] // 100)
            display_matrix = attention_matrix[::stride, ::stride]
        else:
            display_matrix = attention_matrix

        im = plt.imshow(display_matrix, cmap='hot', aspect='auto')
        plt.colorbar(im, label='Attention Score')
        plt.title(title)
        plt.xlabel('Patch Index')
        plt.ylabel('Patch Index')

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f'{title.replace(" ", "_")}_{timestamp}.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path