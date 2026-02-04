# wsi_gene_dataset.py
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import glob
import random
import json
import numpy as np
import pandas as pd


class MyDataSet(Dataset):


    def __init__(self, wsi_feature_dir, gene_feature_dir, cox_txt_path,
                 clinical_data_path=None, hypoxia_pathways_path=None,
                 max_dim=10, mode='train', transform=None,target_patches=300):
        self.wsi_feature_dir = wsi_feature_dir
        self.gene_feature_dir = gene_feature_dir
        self.cox_txt_path = cox_txt_path
        self.clinical_data_path = clinical_data_path
        self.hypoxia_pathways_path = hypoxia_pathways_path
        self.max_dim = max_dim
        self.mode = mode
        self.transform = transform

        print(f"\n=== Dataset初始化 ({mode}) ===")
        print(f"WSI特征目录: {wsi_feature_dir}")
        print(f"基因特征目录: {gene_feature_dir}")
        print(f"生存数据文件: {cox_txt_path}")
        print(f"临床数据路径: {clinical_data_path}")
        print(f"缺氧通路路径: {hypoxia_pathways_path}")

        # 获取所有文件
        self.gene_file_list = glob.glob(os.path.join(self.gene_feature_dir, '*.txt'))
        self.wsi_file_list = glob.glob(os.path.join(self.wsi_feature_dir, '*.pt'))

        print(f"找到基因文件: {len(self.gene_file_list)} 个")
        print(f"找到WSI文件: {len(self.wsi_file_list)} 个")

        if self.gene_file_list:
            print(f"基因文件示例: {os.path.basename(self.gene_file_list[0])}")
        if self.wsi_file_list:
            print(f"WSI文件示例: {os.path.basename(self.wsi_file_list[0])}")


        self._build_patient_maps()


        self.clinical_data = None
        self.clinical_dim = 0
        if clinical_data_path and os.path.exists(clinical_data_path):
            self._load_clinical_data()
        else:
            print("未提供临床数据路径或文件不存在")


        self.hypoxia_data = None
        self.hypoxia_dim = 0
        if hypoxia_pathways_path and os.path.exists(hypoxia_pathways_path):
            self._load_hypoxia_data()
        else:
            print("未提供缺氧通路数据路径或文件不存在")

        self._pre_process()


        print(f"\n=== 数据集统计 ===")
        print(f"总样本数: {len(self.patient_list)}")
        print(f"临床特征维度: {self.clinical_dim}")
        print(f"缺氧通路维度: {self.hypoxia_dim}")

    def _build_patient_maps(self):

        self.gene_patient_to_file = {}
        self.wsi_patient_to_files = {}


        for file_path in self.gene_file_list:
            filename = os.path.basename(file_path)

            try:

                patient_id = filename.split('_')[0][:12]
            except:
                patient_id = filename[:12]
            self.gene_patient_to_file[patient_id] = file_path


        for file_path in self.wsi_file_list:
            filename = os.path.basename(file_path)
            try:
                patient_id = filename.split('_')[0][:12]
            except:
                patient_id = filename[:12]
            if patient_id not in self.wsi_patient_to_files:
                self.wsi_patient_to_files[patient_id] = []
            self.wsi_patient_to_files[patient_id].append(file_path)

        print(f"\n患者映射统计:")
        print(f"  基因患者数: {len(self.gene_patient_to_file)}")
        print(f"  WSI患者数: {len(self.wsi_patient_to_files)}")

    def _load_clinical_data(self):

        try:
            print(f"\n加载临床数据: {self.clinical_data_path}")


            if self.clinical_data_path.endswith('.csv'):
                self.clinical_data = pd.read_csv(self.clinical_data_path)
            elif self.clinical_data_path.endswith('.txt'):
                self.clinical_data = pd.read_csv(self.clinical_data_path, sep='\t')
            elif self.clinical_data_path.endswith(('.xlsx', '.xls')):
                self.clinical_data = pd.read_excel(self.clinical_data_path)
            else:

                self.clinical_data = pd.read_csv(self.clinical_data_path, sep=None, engine='python')


            patient_id_col = self.clinical_data.columns[0]

            self.clinical_data[patient_id_col] = self.clinical_data[patient_id_col].astype(str)

            self.clinical_data[patient_id_col] = self.clinical_data[patient_id_col].apply(
                lambda x: x.split('.')[0] if '.' in x else x
            )

            self.clinical_data[patient_id_col] = self.clinical_data[patient_id_col].apply(
                lambda x: x[:12] if len(x) >= 12 else x
            )


            self.clinical_data.set_index(patient_id_col, inplace=True)


            self.clinical_data = self.clinical_data.fillna(0)


            numeric_cols = self.clinical_data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.clinical_data.select_dtypes(exclude=[np.number]).columns


            if len(categorical_cols) > 0:
                print(f"  对分类变量进行one-hot编码: {list(categorical_cols)}")
                self.clinical_data = pd.get_dummies(self.clinical_data, columns=categorical_cols, drop_first=True)

            self.clinical_dim = self.clinical_data.shape[1]




            numeric_cols = self.clinical_data.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:

                self.original_stats = {
                    col: {
                        'min': self.clinical_data[col].min(),
                        'max': self.clinical_data[col].max(),
                        'mean': self.clinical_data[col].mean(),
                        'std': self.clinical_data[col].std()
                    }
                    for col in numeric_cols
                }


                for col in numeric_cols:
                    col_min = self.clinical_data[col].min()
                    col_max = self.clinical_data[col].max()


                    if col_max > col_min:
                        self.clinical_data[col] = (self.clinical_data[col] - col_min) / (col_max - col_min)
                    else:

                        self.clinical_data[col] = 0.5

                print(f"  已完成Min-Max归一化，处理特征数: {len(numeric_cols)}")

            self.clinical_dim = self.clinical_data.shape[1]

        except Exception as e:
            print(f"加载临床数据失败: {e}")
            self.clinical_data = None
            self.clinical_dim = 0

    def _load_hypoxia_data(self):

        try:
            print(f"\n加载缺氧通路数据: {self.hypoxia_pathways_path}")


            if self.hypoxia_pathways_path.endswith('.csv'):
                self.hypoxia_data = pd.read_csv(self.hypoxia_pathways_path)
            elif self.hypoxia_pathways_path.endswith('.txt'):
                self.hypoxia_data = pd.read_csv(self.hypoxia_pathways_path, sep='\t')
            elif self.hypoxia_pathways_path.endswith(('.xlsx', '.xls')):
                self.hypoxia_data = pd.read_excel(self.hypoxia_pathways_path)
            else:
                self.hypoxia_data = pd.read_csv(self.hypoxia_pathways_path, sep=None, engine='python')


            patient_id_col = self.hypoxia_data.columns[0]


            self.hypoxia_data[patient_id_col] = self.hypoxia_data[patient_id_col].astype(str)

            self.hypoxia_data[patient_id_col] = self.hypoxia_data[patient_id_col].apply(
                lambda x: x.split('.')[0] if '.' in x else x
            )

            self.hypoxia_data[patient_id_col] = self.hypoxia_data[patient_id_col].apply(
                lambda x: x[:12] if len(x) >= 12 else x
            )


            self.hypoxia_data.set_index(patient_id_col, inplace=True)


            self.hypoxia_data = self.hypoxia_data.fillna(0)


            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.hypoxia_data[:] = scaler.fit_transform(self.hypoxia_data)

            self.hypoxia_dim = self.hypoxia_data.shape[1]



        except Exception as e:
            print(f"加载缺氧通路数据失败: {e}")
            self.hypoxia_data = None
            self.hypoxia_dim = 0

    def _pre_process(self):

        print(f"\n=== 查找共同患者 ===")

        with open(self.cox_txt_path) as f:
            cox_time_list = f.read().splitlines()

        self.cox_dict = {}
        cox_patients = []

        for cox_time in cox_time_list:
            if '\t' in cox_time:
                tcga_name, futime, fustat = cox_time.split('\t')

                tcga_name = tcga_name.split('.')[0] if '.' in tcga_name else tcga_name
                tcga_name = tcga_name[:12]  # 确保12位格式
                cox_patients.append(tcga_name)
                self.cox_dict[tcga_name] = [futime, fustat]

        print(f"cox患者数: {len(cox_patients)}")
        print(f"cox患者示例: {cox_patients[:5]}")


        cox_set = set(cox_patients)
        gene_set = set(self.gene_patient_to_file.keys())
        wsi_set = set(self.wsi_patient_to_files.keys())


        clinical_set = set()
        if self.clinical_data is not None:
            clinical_set = set(self.clinical_data.index)
            print(f"  临床患者集合: {len(clinical_set)}")


        hypoxia_set = set()
        if self.hypoxia_data is not None:
            hypoxia_set = set(self.hypoxia_data.index)
            print(f"  缺氧通路患者集合: {len(hypoxia_set)}")


        common_set = cox_set.intersection(gene_set).intersection(wsi_set)
        print(f"\ncox+基因+WSI交集: {len(common_set)}")


        if clinical_set:
            common_set = common_set.intersection(clinical_set)
            print(f"+临床数据交集: {len(common_set)}")


        if hypoxia_set:
            common_set = common_set.intersection(hypoxia_set)
            print(f"+缺氧通路数据交集: {len(common_set)}")

        self.patient_list = list(common_set)

        if len(self.patient_list) > 0:
            print(f"\n 成功找到 {len(self.patient_list)} 个共同患者")
            print(f"共同患者示例: {self.patient_list[:10]}")

            # 统计缺失的患者
            missing_in_clinical = len(cox_set) - len(clinical_set) if clinical_set else 0
            missing_in_hypoxia = len(cox_set) - len(hypoxia_set) if hypoxia_set else 0

            if missing_in_clinical > 0:
                print(f"  警告: {missing_in_clinical} 个患者没有临床数据")
            if missing_in_hypoxia > 0:
                print(f"  警告: {missing_in_hypoxia} 个患者没有缺氧通路数据")

        else:
            print(f"\n 错误：没有找到共同患者！")
            print("请检查:")
            print("1. 患者ID格式是否一致")
            print("2. 文件命名规则")
            print("3. 数据完整性")
            print("\n尝试手动匹配示例:")
            print(f"  cox患者示例: {list(cox_set)[:3]}")
            print(f"  基因患者示例: {list(gene_set)[:3]}")
            print(f"  WSI患者示例: {list(wsi_set)[:3]}")
            if clinical_set:
                print(f"  临床患者示例: {list(clinical_set)[:3]}")
            if hypoxia_set:
                print(f"  缺氧通路患者示例: {list(hypoxia_set)[:3]}")

        if self.mode == 'train' and self.patient_list:
            random.shuffle(self.patient_list)

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, index):
        patient_name = self.patient_list[index]


        gene_file = self.gene_patient_to_file[patient_name]
        with open(gene_file, 'r') as f_gene:
            gene_feat = np.array(json.load(f_gene))


        wsi_files = self.wsi_patient_to_files[patient_name]


        selected_wsi_file = random.choice(wsi_files)


        wsi_tensor = torch.load(selected_wsi_file)


        if isinstance(wsi_tensor, torch.Tensor):
            wsi_feat = wsi_tensor.numpy()
        else:
            wsi_feat = np.array(wsi_tensor)


        target_patches = 500
        if len(wsi_feat) > target_patches:

            indices = np.random.choice(len(wsi_feat), target_patches, replace=False)
            wsi_feat = wsi_feat[indices]
        elif len(wsi_feat) < target_patches:

            padding = np.zeros((target_patches - len(wsi_feat), wsi_feat.shape[1]))
            wsi_feat = np.vstack([wsi_feat, padding])


        clinical_feat = np.zeros(self.clinical_dim, dtype=np.float32)
        if self.clinical_data is not None and patient_name in self.clinical_data.index:
            try:
                clinical_feat = self.clinical_data.loc[patient_name].values.astype(np.float32)

                if len(clinical_feat) != self.clinical_dim:

                    clinical_feat = np.zeros(self.clinical_dim, dtype=np.float32)
            except Exception as e:

                clinical_feat = np.zeros(self.clinical_dim, dtype=np.float32)

        # 获取缺氧通路数据
        hypoxia_feat = np.zeros(self.hypoxia_dim, dtype=np.float32)
        if self.hypoxia_data is not None and patient_name in self.hypoxia_data.index:
            try:
                hypoxia_feat = self.hypoxia_data.loc[patient_name].values.astype(np.float32)

                if len(hypoxia_feat) != self.hypoxia_dim:

                    hypoxia_feat = np.zeros(self.hypoxia_dim, dtype=np.float32)
            except Exception as e:

                hypoxia_feat = np.zeros(self.hypoxia_dim, dtype=np.float32)


        if index < 2:
            print(f"\n[DEBUG] 样本 {index} - 患者 {patient_name}:")
            print(f"  WSI特征形状: {wsi_feat.shape}")
            print(f"  基因特征形状: {gene_feat.shape}")
            print(f"  临床特征形状: {clinical_feat.shape} (前5个值: {clinical_feat[:5]})")
            print(f"  缺氧通路特征形状: {hypoxia_feat.shape} (前5个值: {hypoxia_feat[:5]})")
            print(f"  生存时间: {self.cox_dict[patient_name][0]}")
            print(f"  生存状态: {self.cox_dict[patient_name][1]}")

        if self.transform is not None:
            wsi_feat = self.transform(wsi_feat)

        # 转换为torch tensor
        wsi_feat = torch.FloatTensor(wsi_feat)
        gene_feat = torch.FloatTensor(gene_feat)
        clinical_feat = torch.FloatTensor(clinical_feat)
        hypoxia_feat = torch.FloatTensor(hypoxia_feat)

        # 获取生存数据
        futime = float(self.cox_dict[patient_name][0])
        fustat = float(self.cox_dict[patient_name][1])

        return wsi_feat, gene_feat, clinical_feat, hypoxia_feat, futime, fustat

    @staticmethod
    def collate_fn(batch):
        # 解包批次数据
        wsi_feat, gene_feat, clinical_feat, hypoxia_feat, futime, fustat = tuple(zip(*batch))

        # 处理WSI特征
        wsi_feat_padded = []
        target_patches = 500

        for feat in wsi_feat:
            patches, dim = feat.shape

            if patches < target_patches:

                padding = torch.zeros(target_patches - patches, dim)
                padded = torch.cat([feat, padding], dim=0)
                wsi_feat_padded.append(padded)
            elif patches > target_patches:

                indices = torch.randperm(patches)[:target_patches]
                padded = feat[indices]
                wsi_feat_padded.append(padded)
            else:
                wsi_feat_padded.append(feat)


        wsi_feat = torch.stack(wsi_feat_padded, dim=0)
        gene_feat = torch.stack(gene_feat, dim=0)
        clinical_feat = torch.stack(clinical_feat, dim=0)
        hypoxia_feat = torch.stack(hypoxia_feat, dim=0)


        wsi_feat = wsi_feat.float()
        gene_feat = gene_feat.float()
        clinical_feat = clinical_feat.float()
        hypoxia_feat = hypoxia_feat.float()


        futime = torch.FloatTensor(futime)
        fustat = torch.FloatTensor(fustat)

        return wsi_feat, gene_feat, clinical_feat, hypoxia_feat, futime, fustat