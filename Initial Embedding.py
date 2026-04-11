# ==================== 在你的SNN模型中添加通路分组 ====================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class PathwayEncoderWithGrouping(nn.Module):
    """在原有SNN基础上加入通路分组初始嵌入"""

    def __init__(self, pathway_dict, all_gene_names):
        super().__init__()
        self.gene_to_idx = {gene: i for i, gene in enumerate(all_gene_names)}
        self.pathway_names = list(pathway_dict.keys())

        print(f"基因总数: {len(all_gene_names)}")
        print(f"通路总数: {len(self.pathway_names)}")

        # 1. 首先构建通路-基因矩阵（用于分组聚合）
        self._build_pathway_gene_matrix(pathway_dict, all_gene_names)

        # 2. SNN层（保持你的原有结构）
        self.pathway_layers = nn.ModuleList()
        self.pathway_indices = []

        for name in self.pathway_names:
            genes = pathway_dict[name]
            indices = [self.gene_to_idx[g] for g in genes]
            self.pathway_indices.append(torch.LongTensor(indices))

            # 你的SNN结构
            in_dim = len(indices)
            self.pathway_layers.append(nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.SELU(),
                nn.AlphaDropout(0.1),
                nn.Linear(128, 64),
                nn.SELU()
            ))

        # 3. 添加KIRC重要通路标识（可选）
        self._identify_kirc_pathways()

    def _build_pathway_gene_matrix(self, pathway_dict, all_gene_names):
        """构建通路-基因关联矩阵"""
        num_genes = len(all_gene_names)
        num_pathways = len(self.pathway_names)

        # 创建通路-基因矩阵 [num_pathways, num_genes]
        pathway_gene_matrix = torch.zeros(num_pathways, num_genes)
        self.pathway_gene_counts = []  # 记录每个通路的基因数

        print("构建通路-基因矩阵...")
        for i, pathway_name in enumerate(self.pathway_names):
            genes = pathway_dict[pathway_name]
            indices = [self.gene_to_idx[g] for g in genes if g in self.gene_to_idx]

            if indices:
                # 在矩阵中标记该通路包含的基因
                pathway_gene_matrix[i, indices] = 1.0
                self.pathway_gene_counts.append(len(indices))
            else:
                self.pathway_gene_counts.append(0)

            # 显示HIF-1等重要通路
            if 'HIF' in pathway_name.upper() or 'HYPOXIA' in pathway_name.upper():
                print(f"  重要通路: {pathway_name} - {len(indices)}个基因")

        # 注册为buffer（不参与训练但保存在模型中）
        self.register_buffer('pathway_gene_matrix', pathway_gene_matrix)

        # 计算每个通路的基因数，用于归一化
        self.register_buffer('pathway_counts',
                             torch.tensor(self.pathway_gene_counts, dtype=torch.float32))

        print(f"通路-基因矩阵构建完成: {pathway_gene_matrix.shape}")

    def _identify_kirc_pathways(self):
        """标识KIRC相关的重要通路"""
        self.kirc_important_pathways = []
        self.hif_pathways = []

        for pathway_name in self.pathway_names:
            upper_name = pathway_name.upper()

            # HIF-1信号通路
            if 'HIF' in upper_name or 'HYPOXIA' in upper_name:
                self.hif_pathways.append(pathway_name)
                if pathway_name not in self.kirc_important_pathways:
                    self.kirc_important_pathways.append(pathway_name)

            # VEGF信号通路
            if 'VEGF' in upper_name or 'ANGIOGENESIS' in upper_name:
                if pathway_name not in self.kirc_important_pathways:
                    self.kirc_important_pathways.append(pathway_name)

            # 其他KIRC相关通路
            kirc_keywords = ['KIDNEY', 'RENAL', 'METABOLISM', 'MTOR', 'PI3K']
            for keyword in kirc_keywords:
                if keyword in upper_name:
                    if pathway_name not in self.kirc_important_pathways:
                        self.kirc_important_pathways.append(pathway_name)
                    break

        print(f"\n识别到 {len(self.kirc_important_pathways)} 个KIRC重要通路")
        print(f"其中 {len(self.hif_pathways)} 个是HIF/缺氧相关通路")
        if self.hif_pathways:
            print(f"HIF通路示例: {self.hif_pathways[:3]}")

    def get_pathway_initial_embeddings(self, x, method='mean'):
        """
        获取通路的初始嵌入（你的核心改进）

        参数:
            x: [batch_size, num_genes] 基因表达矩阵
            method: 'mean' - 通路内基因求均值
                   'max' - 通路内基因取最大值
                   'both' - 同时使用均值和最大值

        返回:
            initial_embeddings: [batch_size, num_pathways, feature_dim]
        """
        batch_size = x.shape[0]
        num_pathways = len(self.pathway_names)

        if method == 'mean':
            # 方法1：通路内基因求均值
            # [batch, genes] × [pathways, genes]^T → [batch, pathways]
            pathway_means = torch.matmul(x, self.pathway_gene_matrix.t())

            # 除以基因数进行归一化（避免不同通路基因数不同）
            pathway_means = pathway_means / (self.pathway_counts.unsqueeze(0) + 1e-8)

            # 扩展为3D [batch, pathways, 1]
            initial_embeddings = pathway_means.unsqueeze(-1)

            print(f"通路均值嵌入: {initial_embeddings.shape}")
            return initial_embeddings

        elif method == 'max':
            # 方法2：通路内基因取最大值
            pathway_maxs = []

            for i in range(num_pathways):
                # 获取该通路的基因掩码
                gene_mask = self.pathway_gene_matrix[i].bool()
                if gene_mask.any():
                    # 提取该通路的基因表达
                    pathway_genes = x[:, gene_mask]
                    # 取最大值
                    max_vals, _ = torch.max(pathway_genes, dim=1)
                    pathway_maxs.append(max_vals.unsqueeze(-1))
                else:
                    # 如果没有基因，用零填充
                    pathway_maxs.append(torch.zeros(batch_size, 1, device=x.device))

            # 拼接 [batch, pathways, 1]
            initial_embeddings = torch.stack(pathway_maxs, dim=1)

            print(f"通路最大值嵌入: {initial_embeddings.shape}")
            return initial_embeddings

        elif method == 'both':
            # 方法3：同时使用均值和最大值
            # 计算均值
            pathway_means = torch.matmul(x, self.pathway_gene_matrix.t())
            pathway_means = pathway_means / (self.pathway_counts.unsqueeze(0) + 1e-8)

            # 计算最大值
            pathway_maxs = []
            for i in range(num_pathways):
                gene_mask = self.pathway_gene_matrix[i].bool()
                if gene_mask.any():
                    pathway_genes = x[:, gene_mask]
                    max_vals, _ = torch.max(pathway_genes, dim=1)
                    pathway_maxs.append(max_vals)
                else:
                    pathway_maxs.append(torch.zeros(batch_size, device=x.device))

            pathway_maxs = torch.stack(pathway_maxs, dim=1)

            # 拼接均值和最大值 [batch, pathways, 2]
            initial_embeddings = torch.stack([pathway_means, pathway_maxs], dim=-1)

            print(f"通路均值+最大值嵌入: {initial_embeddings.shape}")
            return initial_embeddings

    def forward(self, x, use_initial_embedding=True, pooling_method='mean'):
        """
        前向传播（修复版）
        """
        batch_size = x.shape[0]

        if use_initial_embedding:
            # 1. 获取通路初始嵌入
            initial_embeddings = self.get_pathway_initial_embeddings(x, method=pooling_method)

            # 2. 对每个通路应用SNN
            pathway_features = []

            for i, layer in enumerate(self.pathway_layers):
                # 提取该通路的原始基因表达
                subset = x[:, self.pathway_indices[i]]

                # 如果该通路有初始嵌入
                if initial_embeddings is not None:
                    # 获取该通路的初始嵌入 [batch, 1] 或 [batch, 2]
                    pathway_init = initial_embeddings[:, i, :]

                    # 方法1：将初始嵌入作为一个单独的特征（不改变基因维度）
                    # 先通过SNN处理基因特征
                    gene_feat = layer(subset)  # [batch, 64]

                    # 然后将初始嵌入与SNN输出结合
                    # 扩展初始嵌入维度以匹配 [batch, 64]
                    if pathway_init.shape[-1] == 1:
                        # 均值嵌入，复制到64维
                        init_expanded = pathway_init.repeat(1, 64)
                    else:
                        # 均值+最大值嵌入 [batch, 2]，需要转换到64维
                        init_expanded = torch.cat([
                            pathway_init[:, 0:1].repeat(1, 32),  # 均值部分
                            pathway_init[:, 1:2].repeat(1, 32)  # 最大值部分
                        ], dim=1)

                    # 结合基因特征和初始嵌入
                    combined_feat = gene_feat + 0.1 * init_expanded  # 加权融合

                    path_feat = combined_feat
                else:
                    # 原始方式
                    path_feat = layer(subset)

                pathway_features.append(path_feat.unsqueeze(1))

            # 3. 拼接所有通路特征
            all_features = torch.cat(pathway_features, dim=1)  # [batch, pathways, 64]

            # 4. KIRC通路加权
            if hasattr(self, 'kirc_important_pathways') and self.kirc_important_pathways:
                weights = torch.ones(len(self.pathway_names), device=x.device)
                for i, pathway_name in enumerate(self.pathway_names):
                    if pathway_name in self.kirc_important_pathways:
                        weights[i] = 1.5

                weighted_features = all_features * weights.unsqueeze(0).unsqueeze(-1)
                all_features = weighted_features

            # 展平
            flattened = all_features.view(batch_size, -1)

            return flattened

        else:
            # 使用原有SNN方式
            embeddings = []
            for i, layer in enumerate(self.pathway_layers):
                subset = x[:, self.pathway_indices[i]]
                path_feat = layer(subset)
                embeddings.append(path_feat.unsqueeze(1))

            pathway_features = torch.cat(embeddings, dim=1)
            flattened = pathway_features.view(batch_size, -1)

            return flattened

    def extract_kirc_pathway_features(self, x):
        """
        专门提取KIRC重要通路的特征
        用于知识引导的门控机制
        """
        if not hasattr(self, 'kirc_important_pathways'):
            return None

        batch_size = x.shape[0]
        features = {}

        # 1. 获取所有通路的初始嵌入（均值）
        all_embeddings = self.get_pathway_initial_embeddings(x, method='mean')

        # 2. 提取KIRC重要通路的特征
        for i, pathway_name in enumerate(self.pathway_names):
            if pathway_name in self.kirc_important_pathways:
                # 该通路的均值表达
                pathway_mean = all_embeddings[:, i, 0]  # [batch]

                # 存储
                features[f'{pathway_name}_mean'] = pathway_mean

        # 3. 计算整体缺氧评分（HIF通路平均）
        if self.hif_pathways:
            hif_scores = []
            for pathway_name in self.hif_pathways:
                idx = self.pathway_names.index(pathway_name)
                hif_scores.append(all_embeddings[:, idx, 0])

            if hif_scores:
                hif_tensor = torch.stack(hif_scores, dim=1)  # [batch, num_hif_pathways]
                features['HIF_pathway_mean'] = torch.mean(hif_tensor, dim=1)
                features['HIF_pathway_max'] = torch.max(hif_tensor, dim=1)[0]

        return features


# ==================== 修改你的prepare_genomic_data函数 ====================
def prepare_genomic_data_with_grouping():
    """准备基因数据，支持通路分组"""

    # 保持你的原有代码
    df_expr = pd.read_csv('KIRC-exprSet.csv', index_col=0)

    print(f"原始数据形状: {df_expr.shape}")

    if df_expr.shape[0] > 10000:
        print("检测到行是基因，列是样本，正在转置...")
        df_expr = df_expr.T

    df_expr = np.log2(df_expr + 1)

    gene_names = df_expr.columns.tolist()
    print(f"\n处理后数据:")
    print(f"样本数: {df_expr.shape[0]}")
    print(f"基因数: {df_expr.shape[1]}")

    # 加载通路（你的原有函数）
    def load_pathways(gmt_path, gene_list):
        pathways = {}
        with open(gmt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                name = parts[0]
                genes = [g for g in parts[2:] if g in gene_list]
                if len(genes) >= 5:
                    pathways[name] = genes
        return pathways

    pathway_dict = load_pathways('c2.cp.kegg_medicus.v2025.1.Hs.symbols.gmt', gene_names)
    print(f"成功加载 {len(pathway_dict)} 个生物学通路。")

    return df_expr, pathway_dict


# ==================== 修改convert_to_pathway_embeddings函数 ====================
def convert_to_pathway_embeddings_with_grouping(model, expression_df,
                                                use_initial_embedding=True,
                                                pooling_method='mean',
                                                batch_size=32,
                                                save_path='pathway_embeddings_grouped.csv'):
    """
    转换函数，支持通路分组嵌入
    """
    model.eval()
    device = next(model.parameters()).device

    expression_tensor = torch.FloatTensor(expression_df.values)
    num_samples = len(expression_df)

    print(f"开始转换 {num_samples} 个样本...")
    print(f"使用初始嵌入: {use_initial_embedding}")
    print(f"池化方法: {pooling_method}")

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="转换进度"):
            batch = expression_tensor[i:i + batch_size].to(device)

            if use_initial_embedding:
                embeddings = model(batch, use_initial_embedding=True,
                                   pooling_method=pooling_method)
            else:
                embeddings = model(batch, use_initial_embedding=False)

            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_np = np.vstack(all_embeddings)

    # 创建特征列名
    num_pathways = len(model.pathway_names)
    feature_columns = []

    for i in range(num_pathways):
        for dim in range(64):  # 每个通路输出64维
            pathway_name_short = model.pathway_names[i].replace(':', '_').replace(' ', '_')
            feature_columns.append(f"{pathway_name_short}_dim{dim}")

    embeddings_df = pd.DataFrame(
        embeddings_np,
        index=expression_df.index,
        columns=feature_columns
    )

    print(f"\n转换完成!")
    print(f"原始基因表达矩阵形状: {expression_df.shape}")
    print(f"通路嵌入矩阵形状: {embeddings_df.shape}")

    # 保存结果
    embeddings_df.to_csv(save_path)
    print(f"结果已保存到: {save_path}")

    return embeddings_df


# ==================== 主函数 ====================
def main_with_grouping():
    """主函数：使用通路分组嵌入"""

    print("=" * 50)
    print("SNN + 通路分组编码 (KEGG知识引导)")
    print("=" * 50)

    # 准备数据
    df_expr, pathway_dict = prepare_genomic_data_with_grouping()

    if df_expr is not None and len(pathway_dict) > 0:
        # 创建改进版模型
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {DEVICE}")

        model = PathwayEncoderWithGrouping(
            pathway_dict=pathway_dict,
            all_gene_names=df_expr.columns.tolist()
        ).to(DEVICE)

        # 测试不同池化方法
        print("\n测试不同池化方法:")

        # 方法1: 均值池化
        print("\n1. 均值池化 (mean pooling):")
        dummy_input = torch.randn(2, len(df_expr.columns)).to(DEVICE)
        output_mean = model(dummy_input, use_initial_embedding=True,
                            pooling_method='mean')
        print(f"  输出形状: {output_mean.shape}")

        # 方法2: 最大值池化
        print("\n2. 最大值池化 (max pooling):")
        output_max = model(dummy_input, use_initial_embedding=True,
                           pooling_method='max')
        print(f"  输出形状: {output_max.shape}")

        # 方法3: 同时使用均值和最大值
        print("\n3. 均值+最大值池化 (both):")
        output_both = model(dummy_input, use_initial_embedding=True,
                            pooling_method='both')
        print(f"  输出形状: {output_both.shape}")

        # 提取KIRC通路特征（用于知识引导）
        print("\n提取KIRC重要通路特征...")
        kirc_features = model.extract_kirc_pathway_features(dummy_input)
        if kirc_features:
            print(f"提取到 {len(kirc_features)} 个KIRC通路特征")
            for key in list(kirc_features.keys())[:5]:
                print(f"  {key}: {kirc_features[key].shape}")

        # 转换所有数据（使用均值池化）
        print("\n" + "=" * 50)
        print("转换所有样本...")

        embeddings_df = convert_to_pathway_embeddings_with_grouping(
            model=model,
            expression_df=df_expr,
            use_initial_embedding=True,
            pooling_method='mean',  # 使用均值池化
            batch_size=32,
            save_path='KIRC_pathway_embeddings_grouped.csv'
        )

        print("\n✅ 处理完成!")
        print(f"最终特征维度: {embeddings_df.shape[1]}")
        print(f"KIRC重要通路数: {len(model.kirc_important_pathways)}")
        print(f"HIF/缺氧通路数: {len(model.hif_pathways)}")

        return embeddings_df, model

    else:
        print("\n错误: 数据准备失败")
        return None, None


# ==================== 运行 ====================
if __name__ == "__main__":
    # 运行改进版
    embeddings_df, model = main_with_grouping()

    if embeddings_df is not None:
        # 打印KIRC重要通路列表
        print("\n" + "=" * 50)
        print("KIRC重要通路列表:")
        print("=" * 50)

        for i, pathway in enumerate(model.kirc_important_pathways[:20]):  # 显示前20个
            print(f"{i + 1}. {pathway}")

        # HIF通路特别标记
        print("\nHIF/缺氧相关通路:")
        for pathway in model.hif_pathways:
            print(f"  - {pathway}")