import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str, eva_imp, entropy_balance

logger = logging.getLogger('CDMOE')

# ========== 新损失函数：路由指导的对比自蒸馏 ==========
def routing_guided_contrastive_loss(student_feat, teacher_feats, weights, temperature=0.1):
    """
    动态路由指导的对比自蒸馏损失 (Routing-guided Contrastive Self-Distillation)
    
    参数:
    student_feat: 融合特征 C [Batch, Dim]
    teacher_feats: 列表，包含独立的单模态特征 [T_l, T_v, T_a]，每个形状为 [Batch, Dim]
    weights: GAP-Router 输出的路由权重 [Batch, 3] (对应 l, v, a)
    temperature: InfoNCE 温度系数
    """
    # 1. 学生特征归一化
    student_feat = F.normalize(student_feat, dim=-1)
    
    total_loss = 0.0
    
    # 遍历三个模态 (Text=0, Vision=1, Audio=2)
    for i, t_feat in enumerate(teacher_feats):
        # 2. 教师特征截断梯度并归一化
        t_feat = t_feat.detach()
        t_feat = F.normalize(t_feat, dim=-1)
        
        # 3. 计算相似度矩阵 [Batch, Batch]
        logits = torch.matmul(student_feat, t_feat.T) / temperature
        
        # 4. 生成对角线正样本标签
        labels = torch.arange(student_feat.size(0)).to(student_feat.device)
        
        # 5. 计算双向 InfoNCE 损失（不降维，保留每个样本的损失）
        loss_s2t = F.cross_entropy(logits, labels, reduction='none')
        loss_t2s = F.cross_entropy(logits.T, labels, reduction='none')
        
        # 该模态的基础对比损失 [Batch]
        base_loss_m = (loss_s2t + loss_t2s) / 2.0
        
        # 6. 获取当前模态的路由权重，并截断梯度
        w_m = weights[:, i].detach()
        
        # 7. 动态权重指导：加权平均得到该模态的最终损失
        weighted_loss_m = torch.mean(w_m * base_loss_m)
        
        total_loss += weighted_loss_m
        
    return total_loss


class CDMOE():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        params = list(model.parameters())
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0

            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision)
                    w = output['channel_weight']  # [batch, 3]

                    y_pred.append(output['logits_c'].cpu())
                    y_true.append(labels.cpu())

                    loss_task_l = self.criterion(output['logits_l'], labels)
                    loss_task_v = self.criterion(output['logits_v'], labels)
                    loss_task_a = self.criterion(output['logits_a'], labels)
                    loss_task_m = self.criterion(output['logits_c'], labels)

                    # 计算各模态与标签的距离/误差
                    l_dist = eva_imp(output['logits_l'], labels)  # [batch] 或 [batch, 1]
                    a_dist = eva_imp(output['logits_a'], labels)  # [batch] 或 [batch, 1]
                    v_dist = eva_imp(output['logits_v'], labels)  # [batch] 或 [batch, 1]

                    # 确保维度一致并正确计算理想权重
                    if l_dist.dim() > 1:
                        l_dist = l_dist.squeeze(-1)
                    if a_dist.dim() > 1:
                        a_dist = a_dist.squeeze(-1)
                    if v_dist.dim() > 1:
                        v_dist = v_dist.squeeze(-1)
                    
                    # 堆叠成 [batch, 3]
                    dists_tensor = torch.stack([l_dist, v_dist, a_dist], dim=1)  # [batch, 3]
                    dists_tensor = dists_tensor + 1e-6  # 防止除零
                    
                    # 计算理想分布：误差越小，权重越大
                    inv_dists = 1.0 / dists_tensor  # [batch, 3]
                    dist_target = inv_dists / inv_dists.sum(dim=1, keepdim=True)  # [batch, 3]，归一化
                    
                    loss_sim = F.mse_loss(w, dist_target.detach())
                    
                    loss_ety = entropy_balance(w)

                    # ========== 路由指导的对比自蒸馏损失 ==========
                    # 构建教师特征列表（顺序必须与 w 的通道一致：text=0, vision=1, audio=2）
                    teacher_feats = [
                        output['l_proj'],  # Text feature
                        output['v_proj'],  # Vision feature
                        output['a_proj']   # Audio feature
                    ]
                    
                    loss_ud = routing_guided_contrastive_loss(
                        student_feat=output['c_proj'],   # 融合特征 C
                        teacher_feats=teacher_feats,     # 单模态特征列表 [T_l, T_v, T_a]
                        weights=w,                        # 动态路由权重 [batch, 3]
                        temperature=0.1
                    )
                    # =============================================

                    loss = loss_task_m + (loss_task_l + loss_task_v + loss_task_a)/3 + \
                           0.1 * (loss_ety + 0.1 * loss_sim) + 0.1 * loss_ud

                    loss.backward()
                    train_loss += loss.item()

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = './pt/emoe.pth'
                torch.save(model.state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False, f=0):
        # 此方法保持原样，无需修改
        model.eval()
        y_pred, y_true = [], []
        weight, ability = [], []
        c_fea = []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    output = model(text, audio, vision)

                    loss = self.criterion(output['logits_c'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['logits_c'].cpu())
                    y_true.append(labels.cpu())
                    
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        
        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
