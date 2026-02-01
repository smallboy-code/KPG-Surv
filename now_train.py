import os
import math
import argparse
import sys
import json

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from regularization import Regularization
# from torchsummary import summary
# from pytorch_summary import torchsummary
from torchinfo import summary

# from my_dataset import MyDataSet
from new_wsi_gene_dataset import MyDataSet as WSI_Gene_DataSet
# from wsi_dataset_cox import MyDataSet as WSI_Dataset
# from vit_model_gene_wsi_concat import my_model as create_model_wsi_gene
from vit_model_knowledge_gated import my_model as create_model_wsi_gene
# from vit_model_gene_wsi_concat_no_contrastive_loss import my_model as create_model_wsi_gene_no_contrastive_loss
# from vit_model_gene_wsi_concat_no_contrastive_loss import my_model as create_model_wsi_gene_no_contrastive_loss
from vit_model_one_cls import my_model as create_model_wsi
from new_utils_cox import read_split_data, train_one_epoch, evaluate
from torch.nn import DataParallel
import shutil
import time
from datetime import datetime

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

        # 加载模型状态
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 处理DataParallel的情况
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

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

    # batch_size = int(args.batch_size / len(args.device.split(',')))
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
    wsi_input_dim = sample_wsi.shape[1]  # 应该是768
    gene_input_dim = sample_gene.shape[1]  # 应该是64

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
    model_log.write('Total params: {:.2f}M\n'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))  # 输出参数数量


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


    optimizer = optim.Adam(pg, lr=args.base_lr, weight_decay=1E-5)  # 使用 base_lr 而不是 lr


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


        (train_loss, train_cox_acc, train_p_value, train_c_index,
         train_preds, train_labels, train_probs) = train_one_epoch(
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
            save_metrics_dir=save_metrics_dir
        )

        scheduler.step()


        (val_loss, val_cox_acc, val_p_value, val_c_index,
         val_preds, val_labels, val_probs) = evaluate(
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
            save_metrics_dir=save_metrics_dir
        )

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



        tb_writer.add_scalar('train_loss', train_loss, epoch)
        tb_writer.add_scalar('train_cox_acc', train_cox_acc, epoch)
        tb_writer.add_scalar('train_p_value', train_p_value, epoch)
        tb_writer.add_scalar('train_c_index', train_c_index, epoch)
        tb_writer.add_scalar('val_loss', val_loss, epoch)
        tb_writer.add_scalar('val_cox_acc', val_cox_acc, epoch)
        tb_writer.add_scalar('val_p_value', val_p_value, epoch)
        tb_writer.add_scalar('val_c_index', val_c_index, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        model_file.write(
            'Train-Epoch-' + str(epoch) + ' : train loss : ' + str(train_loss) + ' ; train cox acc : ' + str(
                train_cox_acc)
            + ' ; train p value : ' + str(train_p_value) + ' ; train c index : ' + str(train_c_index) + '\n')
        model_file.write(
            'Valid-Epoch-' + str(epoch) + ' : valid loss : ' + str(val_loss) + ' ; valid cox acc : ' + str(val_cox_acc)
            + ' ; valid p value : ' + str(val_p_value) + ' ; valid c index : ' + str(val_c_index) + '\n')
        model_file.write(
            'lrlrl-Epoch-' + str(epoch) + ' : learning rate : ' + str(optimizer.param_groups[0]["lr"]) + '\n')
        model_file.flush()

        if args.save_metrics and epoch == best_epoch_for_roc:
            os.makedirs(metrics_dir, exist_ok=True)

            if best_train_probs_for_roc is not None and best_train_labels_for_roc is not None:
                try:
                    from new_utils_cox import save_roc_curve, save_confusion_matrix


                    train_roc_auc = save_roc_curve(
                        best_train_labels_for_roc,
                        best_train_probs_for_roc,
                        epoch,
                        metrics_dir,
                        prefix='train'
                    )
                    print(
                        f"✓ 保存最佳训练ROC曲线（第{epoch}轮，C-index: {best_train_cindex_for_roc:.4f}），AUC: {train_roc_auc:.4f}")


                    train_accuracy, train_cm = save_confusion_matrix(
                        best_train_labels_for_roc,
                        best_train_probs_for_roc,
                        epoch,
                        metrics_dir,
                        prefix='train'
                    )
                    print(f"✓ 保存最佳训练混淆矩阵（第{epoch}轮），准确率: {train_accuracy:.4f}")


                    best_train_file = os.path.join(metrics_dir, 'best_train_metrics.json')
                    with open(best_train_file, 'w') as f:
                        json.dump({
                            'epoch': epoch,
                            'c_index': float(best_train_cindex_for_roc),
                            'roc_auc': float(train_roc_auc),
                            'accuracy': float(train_accuracy),
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }, f, indent=2)

                except Exception as e:
                    print(f"保存最佳训练指标失败: {e}")


            if best_val_probs_for_roc is not None and best_val_labels_for_roc is not None:
                try:
                    from new_utils_cox import save_roc_curve, save_confusion_matrix


                    val_roc_auc = save_roc_curve(
                        best_val_labels_for_roc,
                        best_val_probs_for_roc,
                        epoch,
                        metrics_dir,
                        prefix='val'
                    )
                    print(
                        f"✓ 保存最佳验证ROC曲线（第{epoch}轮，C-index: {best_val_cindex_for_roc:.4f}），AUC: {val_roc_auc:.4f}")

                    # 验证集混淆矩阵
                    val_accuracy, val_cm = save_confusion_matrix(
                        best_val_labels_for_roc,
                        best_val_probs_for_roc,
                        epoch,
                        metrics_dir,
                        prefix='val'
                    )
                    print(f"✓ 保存最佳验证混淆矩阵（第{epoch}轮），准确率: {val_accuracy:.4f}")


                    best_val_file = os.path.join(metrics_dir, 'best_val_metrics.json')
                    with open(best_val_file, 'w') as f:
                        json.dump({
                            'epoch': epoch,
                            'c_index': float(best_val_cindex_for_roc),
                            'roc_auc': float(val_roc_auc),
                            'accuracy': float(val_accuracy),
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }, f, indent=2)

                except Exception as e:
                    print(f"保存最佳验证指标失败: {e}")
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
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--topK', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    #parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--base_lr', type=float, default=0.0001,
                        help='base/peak learning rate')
    parser.add_argument('--warmup_lr', type=float, default=0.00001,
                        help='warmup starting learning rate')
    parser.add_argument('--min_lr', type=float, default=0.000001,
                        help='minimum learning rate at the end of cosine annealing')
    parser.add_argument('--warmup_epochs', type=int, default=20,
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
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\clinical_features.csv",
                        help='Path to clinical data CSV file')

    parser.add_argument('--hypoxia_pathways_path', type=str,
                        default=r"E:\TCGA-KIRC\PAMT-main\wsi_data_process\hypoxia_features\hypoxia_features.csv",
                        help='Path to hypoxia pathways data CSV file')
    parser.add_argument('--save_metrics', type=bool, default=True,
                        help='')
    parser.add_argument('--use_gating', type=bool, default=False,
                  help='是否使用门控机制')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='7,6,5,4', type=str, help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)