import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset

# ==================== 配置 ====================
EXPR_PATH = 'KIRC-exprSet.csv'
# 这里的 GMT 文件定义了哪些基因属于哪个通路，可以在 MSigDB 下载
GMT_PATH = 'c2.cp.kegg_medicus.v2025.1.Hs.symbols.gmt'

OUTPUT_DIR = 'pathway_embeddings_results'
# =============================================

# 1. 通路数据读取函数
def load_pathways(gmt_path, gene_list):
    """读取GMT文件并过滤出存在于exprSet中的基因"""
    pathways = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            name = parts[0]
            # 过滤：只保留存在于我们 exprSet 列名中的基因
            genes = [g for g in parts[2:] if g in gene_list]
            if len(genes) >= 5:  # 忽略基因太少的通路
                pathways[name] = genes
    return pathways


# 2. SNN (Self-Normalizing Network) 编码器模块
class PathwayEncoder(nn.Module):
    def __init__(self, pathway_dict, all_gene_names):
        super().__init__()
        self.gene_to_idx = {gene: i for i, gene in enumerate(all_gene_names)}
        self.pathway_names = list(pathway_dict.keys())

        self.pathway_layers = nn.ModuleList()
        self.pathway_indices = []

        for name in self.pathway_names:
            genes = pathway_dict[name]
            indices = [self.gene_to_idx[g] for g in genes]
            self.pathway_indices.append(torch.LongTensor(indices))

            # SNN 结构：Linear + SELU
            # 这里的输入维度是通路的基因数，输出固定为 128
            in_dim = len(indices)
            self.pathway_layers.append(nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.SELU(),  # 自归一化核心
                nn.AlphaDropout(0.1),
                nn.Linear(128, 64),
                nn.SELU()
            ))

    def forward(self, x):
        # x: [Batch, Total_Genes]
        embeddings = []
        for i, layer in enumerate(self.pathway_layers):
            # 提取该通路的基因子集
            subset = x[:, self.pathway_indices[i]]
            # 编码该通路
            path_feat = layer(subset)
            embeddings.append(path_feat.unsqueeze(1))  # [Batch, 1, 64]

        # 返回通路特征矩阵: [Batch, Num_Pathways, 64]
        return torch.cat(embeddings, dim=1)


# 3. 数据加载与对齐示例
def prepare_genomic_data():
    # 读取基因表达数据
    df_expr = pd.read_csv(EXPR_PATH, index_col=0)

    print(f"原始数据形状: {df_expr.shape}")
    print(f"原始行名示例: {df_expr.index.tolist()[:5]}")
    print(f"原始列名示例: {df_expr.columns.tolist()[:5]}")

    # 检查哪一维是基因
    # 如果行数是19000+，很可能是基因（人类基因组约20000个基因）
    # 如果列数是19000+，那么列是基因

    # 情况1：行是基因，列是样本（需要转置）
    if df_expr.shape[0] > 10000:  # 行数很大，应该是基因
        print("检测到行是基因，列是样本，正在转置...")
        df_expr = df_expr.T  # 转置
        print(f"转置后形状: {df_expr.shape}")

    # 对基因表达进行 Log2 转换
    df_expr = np.log2(df_expr + 1)

    gene_names = df_expr.columns.tolist()
    print(f"\n处理后数据:")
    print(f"形状: {df_expr.shape}")
    print(f"样本数: {df_expr.shape[0]}")
    print(f"基因数: {df_expr.shape[1]}")
    print(f"前10个基因名: {gene_names[:10]}")

    # 建立通路映射
    pathway_dict = load_pathways(GMT_PATH, gene_names)
    print(f"\n成功加载 {len(pathway_dict)} 个生物学通路。")

    if len(pathway_dict) == 0:
        print("\n继续调试基因匹配...")
        # 检查基因名格式问题
        print(f"基因表达数据中基因名示例: {gene_names[:3]}")
        print(f"GMT文件中基因名示例: {['AS3MT', 'COX4I1', 'COX4I2']}")

        # 检查是否有大小写问题
        gmt_genes_lower = [g.lower() for g in ['AS3MT', 'COX4I1', 'COX4I2']]
        expr_genes_lower = [g.lower() for g in gene_names[:50]]

        matches = [g for g in gmt_genes_lower if g in expr_genes_lower]
        print(f"忽略大小写后的匹配: {len(matches)}/{len(gmt_genes_lower)}")

    return df_expr, pathway_dict

# 测试运行
# df_expr, pathway_dict = prepare_genomic_data()
# model = PathwayEncoder(pathway_dict, df_expr.columns)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df_expr, pathway_dict = prepare_genomic_data()
if df_expr is not  None:
    model = PathwayEncoder(pathway_dict,df_expr.columns).to(DEVICE)
    dummy_input = torch.randn(4,len(df_expr.columns)).to(DEVICE)
    output = model(dummy_input)
    print("编码输出形状：", output.shape)


def save_pathways_info(pathway_dict, filename='pathways_info.csv'):
    """保存通路的基本信息到CSV文件"""
    import pandas as pd

    data = []
    for pathway_name, genes in pathway_dict.items():
        data.append({
            'pathway_name': pathway_name,
            'gene_count': len(genes),
            'genes': ';'.join(genes)
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"已保存通路信息到 {filename}，共 {len(df)} 个通路")
    return df


# 使用示例
save_pathways_info(pathway_dict, 'loaded_pathways_info.csv')


# 4. 转换函数
def convert_to_pathway_embeddings(model, expression_df, batch_size=32,
                                  save_path='pathway_embeddings.csv'):
    """
    将基因表达矩阵转换为通路嵌入向量

    参数:
        model: PathwayEncoder模型
        expression_df: 基因表达DataFrame (样本×基因)
        batch_size: 批处理大小
        save_path: 保存路径（可选）

    返回:
        embeddings_df: 通路嵌入向量DataFrame (样本×通路特征)
        pathway_names: 通路名称列表
    """
    model.eval()
    device = next(model.parameters()).device

    # 转换为tensor
    expression_tensor = torch.FloatTensor(expression_df.values)

    # 获取通路名称
    pathway_names = model.pathway_names
    num_pathways = len(pathway_names)
    num_samples = len(expression_df)

    # 将所有信息一次性打印
    print(f"开始转换 {num_samples} 个样本...")
    print(f"输入维度: {expression_tensor.shape[1]} 个基因")
    print(f"输出维度: {num_pathways} 个通路 × 64 维特征")
    print(f"批处理大小: {batch_size}")

    # 分批处理
    all_embeddings = []

    with torch.no_grad():
        # 使用 tqdm 显示进度
        for i in tqdm(range(0, num_samples, batch_size), desc="转换进度"):
            batch = expression_tensor[i:i + batch_size].to(device)

            # 获取通路嵌入
            # pathway_embeddings shape: [batch, num_pathways, 64]
            pathway_embeddings = model(batch)

            # 展平为 [batch, num_pathways * 64]
            batch_size_current = pathway_embeddings.shape[0]
            flattened = pathway_embeddings.view(batch_size_current, -1)

            all_embeddings.append(flattened.cpu().numpy())

    # 合并所有批次
    embeddings_np = np.vstack(all_embeddings)

    # 创建特征列名
    feature_columns = []
    for pathway in pathway_names:
        for dim in range(64):
            feature_columns.append(f"{pathway}_dim{dim}")

    # 创建DataFrame
    embeddings_df = pd.DataFrame(
        embeddings_np,
        index=expression_df.index,  # 保持样本ID
        columns=feature_columns
    )

    print(f"\n转换完成!")
    print(f"原始基因表达矩阵形状: {expression_df.shape}")
    print(f"通路嵌入矩阵形状: {embeddings_df.shape}")

    # 保存结果
    if save_path:
        if save_path.endswith('.h5'):
            embeddings_df.to_hdf(save_path, key='pathway_embeddings')
        elif save_path.endswith('.csv'):
            embeddings_df.to_csv(save_path)
        elif save_path.endswith('.parquet'):
            embeddings_df.to_parquet(save_path)
        elif save_path.endswith('.npy'):
            np.save(save_path, embeddings_np)
            # 单独保存列名
            np.save(save_path.replace('.npy', '_columns.npy'), feature_columns)
            np.save(save_path.replace('.npy', '_pathway_names.npy'), pathway_names)
        else:
            # 默认保存为CSV
            embeddings_df.to_csv(save_path)

        print(f"结果已保存到: {save_path}")

    return embeddings_df, pathway_names


# 5. 保存通路信息的函数
def save_pathways_info(pathway_dict, filename='pathways_info.csv'):
    """保存通路的基本信息到CSV文件"""
    data = []
    for pathway_name, genes in pathway_dict.items():
        data.append({
            'pathway_name': pathway_name,
            'gene_count': len(genes),
            'genes': ';'.join(genes)
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"已保存通路信息到 {filename}，共 {len(df)} 个通路")
    return df



print("KIRC基因表达值到通路嵌入向量转换")

# 1. 准备数据
print("\n步骤1: 准备数据")
df_expr, pathway_dict = prepare_genomic_data()

if df_expr is not None and len(pathway_dict) > 0:
    # 2. 创建模型
    print("\n步骤2: 创建SNN模型")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")

    gene_names = df_expr.columns.tolist()
    model = PathwayEncoder(pathway_dict, gene_names).to(DEVICE)

    # 3. 测试模型
    print("\n步骤3: 测试模型")
    dummy_input = torch.randn(4, len(gene_names)).to(DEVICE)
    output = model(dummy_input)
    print(f"测试输出形状: {output.shape}")

    # 4. 保存通路信息
    print("\n步骤4: 保存通路信息")
    save_pathways_info(pathway_dict, 'loaded_pathways_info.csv')

    # 5. 转换为通路嵌入向量

    print("步骤5: 转换为通路嵌入向量")


    # 直接调用转换函数
    embeddings_df, pathway_names = convert_to_pathway_embeddings(
        model=model,
        expression_df=df_expr,
        batch_size=32,
        save_path='KIRC_pathway_embeddings.csv'
    )

    # 6. 显示结果
    print("转换完成！结果摘要")
    print(f"✓ 样本数量: {embeddings_df.shape[0]}")
    print(f"✓ 通路数量: {len(pathway_names)}")
    print(f"✓ 每个通路嵌入维度: 64")
    print(f"✓ 总特征数量: {embeddings_df.shape[1]}")
    print(f"✓ 结果文件: KIRC_pathway_embeddings.csv")

    # 7. 预览结果
    print(f"\n前3个样本ID:")
    for sample_id in embeddings_df.index[:3]:
        print(f"  {sample_id}")

    print(f"\n前5个特征列:")
    for col in embeddings_df.columns[:5]:
        print(f"  {col}")

    print(f"\n特征矩阵统计:")
    print(f"  均值: {embeddings_df.values.mean():.6f}")
    print(f"  标准差: {embeddings_df.values.std():.6f}")
    print(f"  最小值: {embeddings_df.values.min():.6f}")
    print(f"  最大值: {embeddings_df.values.max():.6f}")


    print("现在您可以使用通路嵌入向量进行后续分析!")


else:
    print("\n错误: 数据准备失败，无法继续转换。")
