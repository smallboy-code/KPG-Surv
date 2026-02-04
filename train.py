import os
import math
import argparse
import sys
import json
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from regularization import Regularization
from torchinfo import summary
from wsi_gene_dataset import MyDataSet as WSI_Gene_DataSet
from vit_model_knowledge_gated import my_model as create_model_wsi_gene
from utils_cox import train_one_epoch, evaluate
from torch.nn import DataParallel
import shutil
import time
from datetime import datetime
import numpy as np


def generate_simple_attention_maps(model, val_loader, epoch, args, is_best_epoch=False):
    """生成简单的注意力热图"""
    if not is_best_epoch:
        return

    print(f"\n{'=' * 60}")
    print(f"为最佳epoch {epoch} 生成注意力热图")
    print(f"{'=' * 60}")

    try:

        from wsi_heatmap_generator_simple import SimpleHeatmapGenerator


        heatmap_dir = os.path.join(args.log_dir, 'attention_heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)

        model.eval()

        for batch_idx, data in enumerate(val_loader):
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


            if len(outputs) < 4:
                print(f"模型输出只有 {len(outputs)} 个元素，需要4个")
                return


            gene2wsi_feature, pred_head, fusion_attn, attention_weights = outputs

            print(f"注意力权重类型: {type(attention_weights)}")
            if attention_weights is not None:
                print(f"注意力权重长度: {len(attention_weights) if isinstance(attention_weights, list) else 'N/A'}")


            generator = SimpleHeatmapGenerator(output_dir=heatmap_dir)


            attention_cpu = generator.tensor_to_cpu_numpy(attention_weights)

            if attention_cpu and len(attention_cpu) > 0:

                first_layer = attention_cpu[0]
                if first_layer is not None:

                    print(f"第一层注意力形状: {first_layer.shape}")

                    if len(first_layer.shape) == 4:

                        sample_attn = first_layer[0]
                        if len(sample_attn.shape) == 3:
                            avg_attn = sample_attn.mean(axis=0)


                            title = f"best_epoch{epoch}_batch{batch_idx}_layer0"
                            heatmap_path = generator.create_attention_heatmap(
                                avg_attn,
                                title,
                                save_path=os.path.join(heatmap_dir, f"{title}.png")
                            )


                            npy_path = os.path.join(heatmap_dir, f"{title}.npy")
                            np.save(npy_path, avg_attn)

                            print(f"✓ 生成热图: {heatmap_path}")
                            print(f"✓ 保存原始数据: {npy_path}")

        print(f"\n注意力热图生成完成！保存到: {heatmap_dir}")

    except Exception as e:
        print(f"生成热图失败: {e}")
        import traceback
        traceback.print_exc()


def create_cosine_scheduler_with_warmup(optimizer, warmup_lr, base_lr, min_lr,
                                        warmup_epochs, total_epochs,
                                        steps_per_epoch):
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            alpha = float(current_step) / float(max(1, warmup_steps))
            current_lr = warmup_lr + alpha * (base_lr - warmup_lr)
            return current_lr / base_lr

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_lr = min_lr + (base_lr - min_lr) * cosine_decay

        return target_lr / base_lr

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model-best.pth')
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if os.path.isfile(checkpoint_path):
        print(f"=> 加载检查点 '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')


        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

            if hasattr(model, 'module'):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)

        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("  优化器状态已加载")

        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("  学习率调度器状态已加载")

        info = {}
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
            print(f"  从第 {checkpoint['epoch']} 轮继续训练")

        if 'best_val_cindex' in checkpoint:
            info['best_val_cindex'] = checkpoint['best_val_cindex']

        if 'best_sum_cindex' in checkpoint:
            info['best_sum_cindex'] = checkpoint['best_sum_cindex']

        return info
    else:
        print(f"=> 未找到检查点 '{checkpoint_path}'")
        return None


def save_all_metrics_to_file(model_file, epoch, train_metrics, val_metrics,
                             train_base=None, val_base=None, lr=None):
    """保存所有指标到文件"""


    train_line = format_metrics_line('Train', epoch, train_metrics, train_base)
    model_file.write(train_line)


    val_line = format_metrics_line('Valid', epoch, val_metrics, val_base)
    model_file.write(val_line)


    if lr is not None:
        model_file.write(f'LR-Epoch-{epoch} : learning rate : {lr:.8f}\n')


    model_file.write('-' * 100 + '\n')
    model_file.flush()


def format_metrics_line(prefix, epoch, metrics_dict, base_metrics=None):

    line_parts = [f'{prefix}-Epoch-{epoch}']


    if base_metrics:
        line_parts.append(f'loss:{base_metrics.get("loss", 0):.6f}')
        line_parts.append(f'cox_acc:{base_metrics.get("cox_acc", 0):.4f}')
        line_parts.append(f'p_value:{base_metrics.get("p_value", 1):.6f}')
        line_parts.append(f'c_index:{base_metrics.get("c_index", 0.5):.4f}')


    if not metrics_dict or not isinstance(metrics_dict, dict):
        return ' : ' + ' ; '.join(line_parts) + '\n'


    actual_metrics = metrics_dict.get('metrics', metrics_dict)


    if '综合指标' in actual_metrics:
        overall = actual_metrics['综合指标']
        line_parts.append(f'综合C指数:{overall.get("综合C指数", 0):.4f}')
        line_parts.append(f'综合Brier:{overall.get("综合Brier评分", 0):.4f}')
        line_parts.append(f'时间AUC:{overall.get("时间AUC", 0):.4f}')


    for time_key in ['1年', '3年', '5年']:
        if time_key in actual_metrics:
            m = actual_metrics[time_key]


            for metric, label in [('c_index', 'Cindex'), ('brier_score', 'Brier'),
                                  ('auc', 'AUC'), ('accuracy', 'Acc')]:
                if metric in m and m[metric] is not None:
                    val = m[metric]
                    if isinstance(val, dict) and 'value' in val:

                        ci_str = ''
                        if 'ci_lower' in val and 'ci_upper' in val:
                            ci_str = f"[{val['ci_lower']:.4f}-{val['ci_upper']:.4f}]"
                        line_parts.append(f'{time_key}_{label}:{val["value"]:.4f}{ci_str}')
                    elif isinstance(val, (int, float)):

                        line_parts.append(f'{time_key}_{label}:{val:.4f}')


            if 'hr' in m and m['hr'] and isinstance(m['hr'], dict):
                hr_info = m['hr']
                if 'value' in hr_info and hr_info['value'] is not None:
                    ci_str = ''
                    if 'ci_lower' in hr_info and 'ci_upper' in hr_info:
                        ci_str = f"[{hr_info['ci_lower']:.4f}-{hr_info['ci_upper']:.4f}]"
                    line_parts.append(f'{time_key}_HR:{hr_info["value"]:.4f}{ci_str}')


            if 'logrank_p' in m and m['logrank_p'] is not None:
                line_parts.append(f'{time_key}_logrank_p:{m["logrank_p"]:.6f}')

    return ' : ' + ' ; '.join(line_parts) + '\n'

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)

    save_metrics_dir = os.path.join(args.log_dir, 'metrics_plots')
    os.makedirs(save_metrics_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    best_val_cindex_for_roc = 0
    best_epoch_for_roc = -1
    best_val_preds_for_roc = None
    best_val_labels_for_roc = None
    best_val_probs_for_roc = None

    best_train_cindex_for_roc = 0
    best_train_preds_for_roc = None
    best_train_labels_for_roc = None
    best_train_probs_for_roc = None
    best_metrics_dict = None
    best_val_metrics = None

    # 保存训练参数
    params_file = os.path.join(args.log_dir, 'training_params.json')
    with open(params_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_dataset = WSI_Gene_DataSet(
        args.wsi_train_feat_dir,
        args.gene_train_feat_dir,
        args.cox_txt_path,
        args.clinical_data_path if hasattr(args, 'clinical_data_path') else None,
        args.hypoxia_pathways_path if hasattr(args, 'hypoxia_pathways_path') else None,
        mode='train'
    )
    print('train patient count: {}'.format(str(len(train_dataset))))

    val_dataset = WSI_Gene_DataSet(
        args.wsi_valid_feat_dir,
        args.gene_valid_feat_dir,
        args.cox_txt_path,
        args.clinical_data_path if hasattr(args, 'clinical_data_path') else None,
        args.hypoxia_pathways_path if hasattr(args, 'hypoxia_pathways_path') else None,
        mode='valid'
    )
    print('valid patient count: {}'.format(str(len(val_dataset))))


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=nw,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             drop_last=False,
                                             collate_fn=val_dataset.collate_fn)

    model_dpr = 0.2
    wsi_block = 6
    clinical_dim = 0
    hypoxia_dim = 0
    if hasattr(train_dataset, 'clinical_dim'):
        clinical_dim = train_dataset.clinical_dim
    if hasattr(train_dataset, 'hypoxia_dim'):
        hypoxia_dim = train_dataset.hypoxia_dim

    print(f"临床特征维度: {clinical_dim}")
    print(f"缺氧通路维度: {hypoxia_dim}")

    sample_wsi, sample_gene, _, _, _, _ = train_dataset[0]
    wsi_input_dim = sample_wsi.shape[1]
    gene_input_dim = sample_gene.shape[1]

    print(f"\n数据维度检查:")
    print(f"  WSI特征维度: {wsi_input_dim}")
    print(f"  基因特征维度: {gene_input_dim}")
    print(f"  WSI patch数: {sample_wsi.shape[0]}")
    print(f"  基因patch数: {sample_gene.shape[0]}")

    if args.train_flag == 0:
        if args.contrastive_loss_flag:
            model = create_model_wsi_gene(
                num_classes=args.num_classes,
                has_logits=False,
                wsi_block=wsi_block,
                gene_block=2,
                dpr=model_dpr,
                clinical_dim=clinical_dim,
                hypoxia_pathways=hypoxia_dim,
                use_context_tokens=args.use_context_tokens,
                embed_wsi_dim=768,
                embed_gene_dim=256,
                wsi_patches=500,
                gene_patches=656,
                gene_input_dim=64,
                use_gating=args.use_gating
            )
        else:
            model = create_model_wsi_gene(
                num_classes=args.num_classes,
                has_logits=False,
                wsi_block=wsi_block,
                gene_block=2,
                dpr=model_dpr,
                clinical_dim=clinical_dim,
                hypoxia_pathways=hypoxia_dim,
                use_context_tokens=args.use_context_tokens,
                embed_wsi_dim=768,
                embed_gene_dim=256,
                wsi_patches=500,
                gene_patches=656,
                gene_input_dim=64
            )

        print("\n[DEBUG] 模型结构信息:")
        print(f"  gene_patches: {656}")
        print(f"  基础特征维度 (gene_patches * 2): {656 * 2} = 1312")
        print(f"  临床特征维度: {clinical_dim}")
        print(f"  缺氧通路维度: {hypoxia_dim}")
        print(f"  使用上下文令牌: {args.use_context_tokens}")

        expected_dim = 1312

        if clinical_dim > 0 and not args.use_context_tokens:
            expected_dim += 32
            print(f"  + 旧临床特征: 32")

        if clinical_dim > 0 and args.use_context_tokens:
            expected_dim += 128
            print(f"  + 上下文令牌特征: 128")

        if hypoxia_dim > 0:
            expected_dim += 16
            print(f"  + 缺氧通路特征: 16")

        print(f"  预期总维度: {expected_dim}")
        print(f"  分类头期望维度: 1456 (从错误信息得知)")
        print(f"  缺失维度: {1456 - expected_dim}")

    shutil.copy(os.path.join(os.getcwd(), sys.argv[0]), args.log_dir)
    model_log_path = os.path.join(args.log_dir, 'model_log.txt')
    model_log = open(model_log_path, 'w')
    model_log.write(str(model))
    model_log.write('\n')
    model_log.write('Total params: {:.2f}M\n'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    print(f"模型配置:")
    print(f"  - WSI Transformer块数: {wsi_block}")
    print(f"  - Gene Transformer块数: 2")
    print(f"  - Drop path rate: {model_dpr}")
    print(f"  - 临床特征维度: {clinical_dim}")
    print(f"  - 缺氧通路维度: {hypoxia_dim}")
    print(f"  - 使用上下文令牌: {args.use_context_tokens}")

    model.cuda()
    if args.train_flag == 0:
        batch_size = 1

        print(f"\n[DEBUG] torchsummary 调用参数:")
        print(f"  clinical_dim: {clinical_dim}")
        print(f"  hypoxia_dim: {hypoxia_dim}")

        wsi_input = torch.randn(batch_size, 500, 768).cuda()
        gene_input = torch.randn(batch_size, 656, 64).cuda()

        input_data = [wsi_input, gene_input]

        if clinical_dim > 0:
            clinical_input = torch.randn(batch_size, clinical_dim).cuda()
            input_data.append(clinical_input)
            print(f"  添加临床数据: {clinical_input.shape}")

        if hypoxia_dim > 0:
            hypoxia_input = torch.randn(batch_size, hypoxia_dim).cuda()
            input_data.append(hypoxia_input)
            print(f"  添加缺氧数据: {hypoxia_input.shape}")

        print(f"  总输入数据长度: {len(input_data)}")

        try:
            print(f"\n[DEBUG] 测试直接调用...")
            if clinical_dim > 0 and hypoxia_dim > 0:
                test_output = model(wsi_input, gene_input, clinical_input, hypoxia_input)
            elif clinical_dim > 0:
                test_output = model(wsi_input, gene_input, clinical_input)
            elif hypoxia_dim > 0:
                test_output = model(wsi_input, gene_input, None, hypoxia_input)
            else:
                test_output = model(wsi_input, gene_input)
            print(f"[DEBUG] 直接调用成功!")

            model_summary = summary(model,
                                    input_data=input_data,
                                    device='cuda',
                                    verbose=0)
            model_log.write(str(model_summary))
            print("[DEBUG] torchsummary 调用成功!")

        except Exception as e:
            print(f"[DEBUG] torchsummary 失败: {e}")

            model_log.write("模型结构:\n")
            model_log.write(f"总参数量: {sum(p.numel() for p in model.parameters()):,}\n")
            model_log.write(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

            # 手动计算分类头维度
            head_dim = 1312
            if clinical_dim > 0 and args.use_context_tokens:
                head_dim += 128
            if hypoxia_dim > 0:
                head_dim += 16

            model_log.write(f"分类头维度: {head_dim} -> {args.num_classes}\n")
            model_log.write("(由于torchsummary失败，使用简化的模型描述)\n")

    elif args.train_flag == 1:
        # 单模态WSI模型输入形状
        model_log.write(str(summary(model, input_size=(1, 500, 768), device='cuda')))
    model_log.close()
    model = DataParallel(model, device_ids=None)
    model = model.cuda()

    # 初始化优化器
    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "head_dist" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    if args.weight_decay > 0:
        regular_loss = Regularization(model, args.weight_decay, p=1).to(device)
    else:
        regular_loss = False

    optimizer = optim.Adam(pg, lr=args.base_lr, weight_decay=1E-5)

    steps_per_epoch = len(train_loader)
    scheduler = create_cosine_scheduler_with_warmup(
        optimizer=optimizer,
        warmup_lr=args.warmup_lr,
        base_lr=args.base_lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch
    )

    start_epoch = 0
    best_val_cindex = 0
    best_sum_cindex = 0

    if args.resume:
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, scheduler)
        if checkpoint_info:
            if 'epoch' in checkpoint_info:
                start_epoch = checkpoint_info['epoch']
            if 'best_val_cindex' in checkpoint_info:
                best_val_cindex = checkpoint_info['best_val_cindex']
            if 'best_sum_cindex' in checkpoint_info:
                best_sum_cindex = checkpoint_info['best_sum_cindex']
            print(
                f"恢复训练参数: epoch={start_epoch}, best_val_cindex={best_val_cindex:.4f}, best_sum_cindex={best_sum_cindex:.4f}")
    elif args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location='cuda')

        if 'state_dict' in weights_dict:
            state_dict = weights_dict['state_dict']
        else:
            state_dict = weights_dict

        if args.finetune:
            del_keys = ['head.weight', 'head.bias'] if hasattr(model.module, 'has_logits') and model.module.has_logits \
                else ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']
            for k in del_keys:
                if k in state_dict:
                    del state_dict[k]
                    print(f"删除分类头权重: {k}")

        print(model.load_state_dict(state_dict, strict=False))
        print(f"预训练权重加载完成")

    if args.num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()

    save_name_txt = os.path.join(args.log_dir, "train_valid_acc.txt")
    model_file = open(save_name_txt, "w")

    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    metrics_dir = os.path.join(args.log_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")

        bootstrap_n = 100 if args.bootstrap else 0

        train_result = train_one_epoch(
            model=model,
            topK=args.topK,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            reg_loss=regular_loss,
            train_flag=args.train_flag,
            contrastive_loss_flag=args.contrastive_loss_flag,
            clinical_dim=clinical_dim,
            hypoxia_dim=hypoxia_dim,
            use_context_tokens=args.use_context_tokens,
            save_metrics_dir=save_metrics_dir,
            is_best_epoch=(epoch == best_epoch_for_roc),
            bootstrap_n=bootstrap_n
        )


        if len(train_result) == 5:
            train_loss, train_metrics, train_preds, train_labels, train_times = train_result
        else:

            train_loss, train_metrics = train_result[:2]
            train_preds = None
            train_labels = None
            train_times = None


        train_cox_acc = 0
        train_p_value = 1.0
        train_c_index = 0.5

        if train_preds is not None and train_labels is not None and train_times is not None:
            try:
                from utils_cox import accuracy_cox, cox_log_rank, CIndex_lifeline


                if train_preds.shape[1] > 1:

                    pred_avg = np.mean(train_preds, axis=1, keepdims=True)
                    train_cox_acc = accuracy_cox(torch.from_numpy(pred_avg), torch.from_numpy(train_labels))
                    train_p_value = cox_log_rank(torch.from_numpy(pred_avg), torch.from_numpy(train_labels),
                                                 torch.from_numpy(train_times))
                    train_c_index = CIndex_lifeline(torch.from_numpy(pred_avg), torch.from_numpy(train_labels),
                                                    torch.from_numpy(train_times))
                else:

                    train_cox_acc = accuracy_cox(torch.from_numpy(train_preds), torch.from_numpy(train_labels))
                    train_p_value = cox_log_rank(torch.from_numpy(train_preds), torch.from_numpy(train_labels),
                                                 torch.from_numpy(train_times))
                    train_c_index = CIndex_lifeline(torch.from_numpy(train_preds), torch.from_numpy(train_labels),
                                                    torch.from_numpy(train_times))

                print(
                    f"[INFO] 训练集基础指标: C-index={train_c_index:.4f}, Cox Acc={train_cox_acc:.4f}, p-value={train_p_value:.6f}")
            except Exception as e:
                print(f"[WARNING] 计算训练集基础指标失败: {e}")


        train_probs = None
        if train_preds is not None:

            if len(train_preds.shape) == 1 or train_preds.shape[1] == 1:

                if len(train_preds.shape) == 1:
                    train_probs = 1 / (1 + np.exp(-train_preds))
                else:
                    train_probs = 1 / (1 + np.exp(-train_preds[:, 0]))
            else:

                train_probs = 1 / (1 + np.exp(-train_preds[:, 0]))


        if train_c_index > best_train_cindex_for_roc and train_preds is not None:
            best_train_cindex_for_roc = train_c_index
            best_train_preds_for_roc = train_preds
            best_train_labels_for_roc = train_labels
            best_train_probs_for_roc = train_probs
            best_train_times_for_roc = train_times
            print(f"更新最佳训练C-index: {train_c_index:.4f} (第{epoch}轮)")

        scheduler.step()

        val_result = evaluate(
            model=model,
            topK=args.topK,
            criterion=criterion,
            data_loader=val_loader,
            epoch=epoch,
            json_path='valid_log.txt',
            reg_loss=regular_loss,
            train_flag=args.train_flag,
            contrastive_loss_flag=args.contrastive_loss_flag,
            clinical_dim=clinical_dim,
            hypoxia_dim=hypoxia_dim,
            use_context_tokens=args.use_context_tokens,
            save_metrics_dir=save_metrics_dir,
            bootstrap_n=bootstrap_n
        )


        if len(val_result) == 9:

            val_loss, val_cox_acc, val_p_value, val_c_index, val_preds, val_labels, val_probs, val_multi_metrics, val_times = val_result
        elif len(val_result) == 8:

            val_loss, val_cox_acc, val_p_value, val_c_index, val_preds, val_labels, val_probs, val_multi_metrics = val_result
            val_times = None
        elif len(val_result) == 7:

            val_loss, val_cox_acc, val_p_value, val_c_index, val_preds, val_labels, val_probs = val_result
            val_multi_metrics = None
            val_times = None
        else:

            val_loss = val_result[0] if len(val_result) > 0 else 0
            val_cox_acc = val_result[1] if len(val_result) > 1 else 0
            val_p_value = val_result[2] if len(val_result) > 2 else 1.0
            val_c_index = val_result[3] if len(val_result) > 3 else 0.5
            val_preds = val_result[4] if len(val_result) > 4 else None
            val_labels = val_result[5] if len(val_result) > 5 else None
            val_probs = val_result[6] if len(val_result) > 6 else None
            val_multi_metrics = None
            val_times = None


            if val_preds is not None:
                if len(val_preds.shape) == 1 or val_preds.shape[1] == 1:
                    if len(val_preds.shape) == 1:
                        val_probs = 1 / (1 + np.exp(-val_preds))
                    else:
                        val_probs = 1 / (1 + np.exp(-val_preds[:, 0]))
                else:
                    val_probs = 1 / (1 + np.exp(-val_preds[:, 0]))


        current_is_best = False
        if val_c_index > best_val_cindex_for_roc:
            current_is_best = True
            print(f" 发现新的最佳epoch: {epoch}, C-index: {val_c_index:.4f}")


        if val_c_index > best_val_cindex_for_roc and val_preds is not None:
            best_val_cindex_for_roc = val_c_index
            best_val_preds_for_roc = val_preds
            best_val_labels_for_roc = val_labels
            best_val_probs_for_roc = val_probs
            best_epoch_for_roc = epoch
            print(f"更新最佳验证C-index用于ROC/CM: {val_c_index:.4f} (第{epoch}轮)")


        if args.save_metrics and current_is_best:
            print("\n" + "=" * 60)
            print(f"生成注意力热图（新的最佳epoch: {epoch}, C-index: {val_c_index:.4f}）")
            print("=" * 60)

            try:
                model.eval()


                for step, data in enumerate(val_loader):
                    if step >= 1:
                        break

                    wsi_features, gene_features, clinical_data, hypoxia_data, futime, fustat = data

                    with torch.no_grad():
                        if clinical_data is not None or hypoxia_data is not None:
                            outputs = model(wsi_features.cuda(), gene_features.cuda(),
                                            clinical_data, hypoxia_data)
                        else:
                            outputs = model(wsi_features.cuda(), gene_features.cuda())


                    if len(outputs) >= 4:
                        gene2wsi_feature, pred_head, fusion_attn, attention_weights = outputs


                        if attention_weights is not None:

                            heatmap_dir = os.path.join(args.log_dir, 'attention_heatmaps')
                            os.makedirs(heatmap_dir, exist_ok=True)


                            save_path = os.path.join(heatmap_dir, f'best_epoch_{epoch}_attention.npy')


                            if isinstance(attention_weights, list) and len(attention_weights) > 0:

                                first_layer = attention_weights[0]
                                if isinstance(first_layer, list) and len(first_layer) > 0:

                                    first_gpu_data = first_layer[0]
                                    if first_gpu_data is not None:

                                        if len(first_gpu_data.shape) == 4:
                                            sample_attn = first_gpu_data[0, 0]
                                            np.save(save_path, sample_attn.cpu().numpy())
                                        else:
                                            np.save(save_path, first_gpu_data.cpu().numpy())
                                        print(f"✓ 保存注意力权重: {save_path}")
                                elif isinstance(first_layer, torch.Tensor):

                                    if len(first_layer.shape) == 4:
                                        sample_attn = first_layer[0, 0]
                                        np.save(save_path, sample_attn.cpu().numpy())
                                    else:
                                        np.save(save_path, first_layer.cpu().numpy())
                                    print(f"✓ 保存注意力权重: {save_path}")

                            print(f"注意力权重已保存到: {save_path}")


                            try:

                                attn_data = np.load(save_path)
                                print(f"注意力矩阵形状: {attn_data.shape}")


                                import matplotlib.pyplot as plt
                                plt.figure(figsize=(10, 8))


                                if attn_data.shape[0] > 100:
                                    stride = max(1, attn_data.shape[0] // 100)
                                    display_data = attn_data[::stride, ::stride]
                                else:
                                    display_data = attn_data

                                plt.imshow(display_data, cmap='hot', aspect='auto')
                                plt.colorbar(label='Attention Score')
                                plt.title(f'Best Epoch {epoch} - Attention Heatmap')
                                plt.xlabel('Patch Index')
                                plt.ylabel('Patch Index')

                                heatmap_path = os.path.join(heatmap_dir, f'best_epoch_{epoch}_heatmap.png')
                                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                                plt.close()
                                print(f"✓ 生成热图: {heatmap_path}")
                            except Exception as e:
                                print(f"生成热图失败: {e}")
            except Exception as e:
                print(f"保存注意力权重失败: {e}")
                import traceback
                traceback.print_exc()

        if train_c_index > best_train_cindex_for_roc and train_preds is not None:
            best_train_cindex_for_roc = train_c_index
            best_train_preds_for_roc = train_preds
            best_train_labels_for_roc = train_labels
            best_train_probs_for_roc = train_probs
            print(f"更新最佳训练C-index用于ROC/CM: {train_c_index:.4f} (第{epoch}轮)")

        if val_c_index > best_val_cindex_for_roc and val_preds is not None:
            best_val_cindex_for_roc = val_c_index
            best_val_preds_for_roc = val_preds
            best_val_labels_for_roc = val_labels
            best_val_probs_for_roc = val_probs
            best_epoch_for_roc = epoch
            print(f"更新最佳验证C-index用于ROC/CM: {val_c_index:.4f} (第{epoch}轮)")

        # 记录到TensorBoard
        tb_writer.add_scalar('train_loss', train_loss, epoch)
        tb_writer.add_scalar('train_c_index', train_c_index, epoch)
        tb_writer.add_scalar('val_loss', val_loss, epoch)
        tb_writer.add_scalar('val_cox_acc', val_cox_acc, epoch)
        tb_writer.add_scalar('val_p_value', val_p_value, epoch)
        tb_writer.add_scalar('val_c_index', val_c_index, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)


        if train_metrics and isinstance(train_metrics, dict):
            for time_key in ['1年', '3年', '5年']:
                if time_key in train_metrics:
                    metrics = train_metrics[time_key]
                    for metric_name in ['c_index', 'brier_score', 'auc', 'accuracy']:
                        if metric_name in metrics:
                            value = metrics[metric_name]
                            if isinstance(value, dict):
                                tb_writer.add_scalar(f'train_{time_key}_{metric_name}', value['value'], epoch)
                            else:
                                tb_writer.add_scalar(f'train_{time_key}_{metric_name}', value, epoch)


        if train_metrics and isinstance(train_metrics, dict):

            if '综合指标' in train_metrics:
                overall = train_metrics['综合指标']
                tb_writer.add_scalar('train/综合C指数', overall.get('综合C指数', 0), epoch)
                tb_writer.add_scalar('train/综合Brier评分', overall.get('综合Brier评分', 0), epoch)
                tb_writer.add_scalar('train/时间AUC', overall.get('时间AUC', 0), epoch)


            for time_key in ['1年', '3年', '5年']:
                if time_key in train_metrics.get('metrics', train_metrics):
                    m = train_metrics.get('metrics', train_metrics)[time_key]
                    for metric_name in ['c_index', 'brier_score', 'auc', 'accuracy']:
                        if metric_name in m and m[metric_name] is not None:
                            val = m[metric_name]
                            if isinstance(val, dict) and 'value' in val:
                                tb_writer.add_scalar(f'train/{time_key}_{metric_name}', val['value'], epoch)
                            elif isinstance(val, (int, float)):
                                tb_writer.add_scalar(f'train/{time_key}_{metric_name}', val, epoch)


        if val_multi_metrics and isinstance(val_multi_metrics, dict):

            if '综合指标' in val_multi_metrics:
                overall = val_multi_metrics['综合指标']
                tb_writer.add_scalar('val/综合C指数', overall.get('综合C指数', 0), epoch)
                tb_writer.add_scalar('val/综合Brier评分', overall.get('综合Brier评分', 0), epoch)
                tb_writer.add_scalar('val/时间AUC', overall.get('时间AUC', 0), epoch)


            for time_key in ['1年', '3年', '5年']:
                if time_key in val_multi_metrics.get('metrics', val_multi_metrics):
                    m = val_multi_metrics.get('metrics', val_multi_metrics)[time_key]
                    for metric_name in ['c_index', 'brier_score', 'auc', 'accuracy']:
                        if metric_name in m and m[metric_name] is not None:
                            val = m[metric_name]
                            if isinstance(val, dict) and 'value' in val:
                                tb_writer.add_scalar(f'val/{time_key}_{metric_name}', val['value'], epoch)
                            elif isinstance(val, (int, float)):
                                tb_writer.add_scalar(f'val/{time_key}_{metric_name}', val, epoch)

        # 保存所有指标到文件
        save_all_metrics_to_file(
            model_file=model_file,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_multi_metrics,
            train_base={
                'loss': train_loss,
                'cox_acc': train_cox_acc,
                'p_value': train_p_value,
                'c_index': train_c_index
            },
            val_base={
                'loss': val_loss,
                'cox_acc': val_cox_acc,
                'p_value': val_p_value,
                'c_index': val_c_index
            },
            lr=optimizer.param_groups[0]["lr"]
        )
        model_file.flush()


        if args.save_metrics and epoch == best_epoch_for_roc:
            os.makedirs(metrics_dir, exist_ok=True)

            # 初始化变量
            train_rocs = None
            train_roc_auc = None
            val_rocs = None
            val_roc_auc = None


            if best_train_probs_for_roc is not None and best_train_labels_for_roc is not None:
                try:
                    from utils_cox import save_roc_curve, save_confusion_matrix


                    if best_train_preds_for_roc is not None and len(best_train_preds_for_roc.shape) >= 2:

                        train_rocs = save_roc_curve(
                            best_train_labels_for_roc,
                            best_train_preds_for_roc,
                            epoch,
                            metrics_dir,
                            prefix='train',
                            keep_latest_only=True
                        )

                        if isinstance(train_rocs, dict):
                            print(f"✓ 保存多时间点训练ROC曲线（第{epoch}轮）:")
                            for time_label, auc_value in train_rocs.items():
                                print(f"  - {time_label} AUC: {auc_value:.4f}")
                        else:
                            train_roc_auc = train_rocs
                            print(f"✓ 保存训练ROC曲线，AUC: {train_roc_auc:.4f}")
                    else:

                        train_roc_auc = save_roc_curve(
                            best_train_labels_for_roc,
                            best_train_probs_for_roc,
                            epoch,
                            metrics_dir,
                            prefix='train'
                        )
                        print(f"✓ 保存训练ROC曲线（第{epoch}轮），AUC: {train_roc_auc:.4f}")


                    if best_train_preds_for_roc is not None and len(best_train_preds_for_roc.shape) >= 2:

                        train_accuracies, train_cm_data = save_confusion_matrix(
                            best_train_labels_for_roc,
                            best_train_preds_for_roc,
                            epoch,
                            metrics_dir,
                            prefix='train',
                            keep_latest_only=True
                        )

                        if isinstance(train_accuracies, dict):
                            print(f"✓ 保存多时间点训练混淆矩阵（第{epoch}轮）:")
                            for time_label, accuracy in train_accuracies.items():
                                print(f"  - {time_label} 准确率: {accuracy:.4f}")
                        else:
                            train_accuracy = train_accuracies
                            print(f"✓ 保存训练混淆矩阵，准确率: {train_accuracy:.4f}")
                    else:

                        train_accuracy, train_cm_data = save_confusion_matrix(
                            best_train_labels_for_roc,
                            best_train_probs_for_roc,
                            epoch,
                            metrics_dir,
                            prefix='train',
                            keep_latest_only=True
                        )
                        print(f"✓ 保存训练混淆矩阵（第{epoch}轮），准确率: {train_accuracy:.4f}")

                except Exception as e:
                    print(f"保存训练集指标失败: {e}")


            if best_val_probs_for_roc is not None and best_val_labels_for_roc is not None:
                try:
                    from utils_cox import save_roc_curve, save_confusion_matrix


                    if best_val_preds_for_roc is not None and len(best_val_preds_for_roc.shape) >= 2:

                        val_rocs = save_roc_curve(
                            best_val_labels_for_roc,
                            best_val_preds_for_roc,
                            epoch,
                            metrics_dir,
                            prefix='val',
                            keep_latest_only=True
                        )

                        if isinstance(val_rocs, dict):
                            print(f"✓ 保存多时间点验证ROC曲线（第{epoch}轮）:")
                            for time_label, auc_value in val_rocs.items():
                                print(f"  - {time_label} AUC: {auc_value:.4f}")
                        else:
                            val_roc_auc = val_rocs
                            print(f"✓ 保存验证ROC曲线，AUC: {val_roc_auc:.4f}")
                    else:

                        val_roc_auc = save_roc_curve(
                            best_val_labels_for_roc,
                            best_val_probs_for_roc,
                            epoch,
                            metrics_dir,
                            prefix='val'
                        )
                        print(f"✓ 保存验证ROC曲线（第{epoch}轮），AUC: {val_roc_auc:.4f}")


                    if best_val_preds_for_roc is not None and len(best_val_preds_for_roc.shape) >= 2:

                        val_accuracies, val_cm_data = save_confusion_matrix(
                            best_val_labels_for_roc,
                            best_val_preds_for_roc,
                            epoch,
                            metrics_dir,
                            prefix='val',
                            keep_latest_only=True
                        )

                        if isinstance(val_accuracies, dict):
                            print(f"✓ 保存多时间点验证混淆矩阵（第{epoch}轮）:")
                            for time_label, accuracy in val_accuracies.items():
                                print(f"  - {time_label} 准确率: {accuracy:.4f}")
                        else:
                            val_accuracy = val_accuracies
                            print(f"✓ 保存验证混淆矩阵，准确率: {val_accuracy:.4f}")
                    else:

                        val_accuracy, val_cm_data = save_confusion_matrix(
                            best_val_labels_for_roc,
                            best_val_probs_for_roc,
                            epoch,
                            metrics_dir,
                            prefix='val',
                            keep_latest_only=True
                        )
                        print(f"✓ 保存验证混淆矩阵（第{epoch}轮），准确率: {val_accuracy:.4f}")

                except Exception as e:
                    print(f"保存验证集指标失败: {e}")


            best_train_file = os.path.join(metrics_dir, 'best_train_metrics.json')
            with open(best_train_file, 'w') as f:
                train_metrics_data = {
                    'epoch': epoch,
                    'c_index': float(best_train_cindex_for_roc),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }


                if train_roc_auc is not None:
                    train_metrics_data['roc_auc'] = float(train_roc_auc)
                elif train_rocs is not None and isinstance(train_rocs, dict):
                    for time_label, auc_value in train_rocs.items():
                        train_metrics_data[f'auc_{time_label}'] = float(auc_value)
                    auc_values = [v for v in train_rocs.values() if isinstance(v, (int, float))]
                    if auc_values:
                        train_metrics_data['auc_mean'] = float(np.mean(auc_values))

                json.dump(train_metrics_data, f, indent=2)
                print(f"✓ 保存训练集最佳指标: {best_train_file}")


            best_val_file = os.path.join(metrics_dir, 'best_val_metrics.json')
            with open(best_val_file, 'w') as f:
                val_metrics_data = {
                    'epoch': epoch,
                    'c_index': float(best_val_cindex_for_roc),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }


                if val_roc_auc is not None:
                    val_metrics_data['roc_auc'] = float(val_roc_auc)
                elif val_rocs is not None and isinstance(val_rocs, dict):
                    for time_label, auc_value in val_rocs.items():
                        val_metrics_data[f'auc_{time_label}'] = float(auc_value)
                    auc_values = [v for v in val_rocs.values() if isinstance(v, (int, float))]
                    if auc_values:
                        val_metrics_data['auc_mean'] = float(np.mean(auc_values))


                if 'val_accuracies' in locals() and isinstance(val_accuracies, dict):
                    for time_label, accuracy in val_accuracies.items():
                        val_metrics_data[f'cm_accuracy_{time_label}'] = float(accuracy)
                    accuracy_values = [v for v in val_accuracies.values() if isinstance(v, (int, float))]
                    if accuracy_values:
                        val_metrics_data['cm_accuracy_mean'] = float(np.mean(accuracy_values))
                elif 'val_accuracy' in locals():
                    val_metrics_data['cm_accuracy'] = float(val_accuracy)

                json.dump(val_metrics_data, f, indent=2)
                print(f"✓ 保存验证集最佳指标: {best_val_file}")
        else:
            if args.save_metrics:
                print(
                    f"当前轮次{epoch}不是最佳，跳过保存ROC/CM（最佳为第{best_epoch_for_roc}轮，C-index: {best_val_cindex_for_roc:.4f}）")

        torch.save(model.state_dict(), os.path.join(args.log_dir, 'model-latest.pth'))

        is_best_val = val_c_index >= best_val_cindex
        is_best_sum = val_c_index + train_c_index >= best_sum_cindex

        if is_best_val:
            best_val_cindex = val_c_index
            model_file.write(f'Epoch {epoch}: 保存最佳验证c-index {best_val_cindex:.4f} 检查点\n')

        if is_best_sum:
            best_sum_cindex = val_c_index + train_c_index
            model_file.write(f'Epoch {epoch}: 保存最佳总c-index {best_sum_cindex:.4f} 检查点\n')

        if is_best_val:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model-val-best.pth'))
            if train_c_index >= 0.9:
                os.rename(os.path.join(args.log_dir, 'model-val-best.pth'),
                          os.path.join(args.log_dir, 'model-val-{}.pth'.format(str(round(val_c_index, 4)))))

        if is_best_sum:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'model-sum-best.pth'))
            if train_c_index >= 0.9:
                os.rename(os.path.join(args.log_dir, 'model-sum-best.pth'),
                          os.path.join(args.log_dir, 'model-sum-{}.pth'.format(str(round(val_c_index, 4)))))

    model_file.close()
    tb_writer.close()

    # 保存最终模型
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
    save_checkpoint({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_cindex': best_val_cindex,
        'best_sum_cindex': best_sum_cindex,
        'args': vars(args)
    }, False, checkpoint_dir, 'final_checkpoint.pth')

    print(f"\n训练完成！")
    print(f"最佳验证c-index: {best_val_cindex:.4f}")
    print(f"最佳总c-index: {best_sum_cindex:.4f}")
    print(f"最终模型已保存至: {final_checkpoint_path}")


if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--topK', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--base_lr', type=float, default=0.0012,
                        help='base/peak learning rate')
    parser.add_argument('--warmup_lr', type=float, default=0.0005,
                        help='warmup starting learning rate')
    parser.add_argument('--min_lr', type=float, default=0.00001,
                        help='minimum learning rate at the end of cosine annealing')
    parser.add_argument('--warmup_epochs', type=int, default=30,
                        help='number of warmup epochs')
    parser.add_argument('--lrf', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--use_context_tokens', type=bool, default=True,
                        help='是否使用Perceiver-style临床上下文令牌')

    parser.add_argument('--resume', type=str, default=r'',
                        help='恢复训练的检查点路径（包含完整训练状态）')
    parser.add_argument('--weights', type=str, default='',
                        help='预训练权重路径（仅模型权重）')
    parser.add_argument('--finetune', type=bool, default=False,
                        help='是否进行微调（加载权重时会删除分类头）')

    log_dir_name = f'log_lusc_norm/{current_time}'

    parser.add_argument('--log_dir', type=str,
                        default=log_dir_name,
                        help='log directory')
    parser.add_argument('--train_flag', type=int, default=0,
                        help='train mode, 0: wsi + gene, 1: wsi, 2: gene')
    parser.add_argument('--contrastive_loss_flag', type=int, default=0,
                        help='train mode, 0: no contrastive loss, 1: contrastive loss')
    parser.add_argument('--cox_txt_path', type=str,
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\cox_time_kirc.txt")

    parser.add_argument('--wsi_train_feat_dir', type=str,
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\wsi_features\train")

    parser.add_argument('--wsi_valid_feat_dir', type=str,
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\wsi_features\valid")

    parser.add_argument('--gene_train_feat_dir', type=str,
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\gene_features\train")

    parser.add_argument('--gene_valid_feat_dir', type=str,
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\gene_features\valid")

    parser.add_argument('--clinical_data_path', type=str,
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\clinical_features1_cleaned.csv",
                        help='Path to clinical data CSV file')

    parser.add_argument('--hypoxia_pathways_path', type=str,
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\hypoxia_features\hypoxia_features.csv",
                        help='Path to hypoxia pathways data CSV file')
    parser.add_argument('--save_metrics', type=bool, default=True,
                        help='')
    parser.add_argument('--use_gating', type=bool, default=True,
                        help='是否使用门控机制')
    parser.add_argument('--bootstrap', type=int, default=100,
                        help='Bootstrap次数（0表示不使用，建议100-200）')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='7,6,5,4', type=str, help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)