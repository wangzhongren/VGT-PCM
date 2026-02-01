# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import json
import os


# ==========================================
# 2. 数据加载器 (保持流式读取)
# ==========================================
class StreamDataset(IterableDataset):
    def __init__(self, file_path, vocab, seq_len=256):
        self.file_path = file_path
        self.vocab = vocab
        self.seq_len = seq_len
    def __iter__(self):
        buf = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "") + " "
                    ids = [self.vocab.get(c, self.vocab["[UNK]"]) for c in text]
                    buf.extend(ids)
                    while len(buf) >= self.seq_len + 1:
                        yield (torch.LongTensor(buf[:self.seq_len]), 
                               torch.LongTensor(buf[1:self.seq_len+1]))
                        buf = buf[self.seq_len:]
                except: continue


# ==========================================
# 1. VGT-Pro 引擎 (支持隐藏层捕获)
# ==========================================
class VGT_ResidualBlock(nn.Module):
    def __init__(self, d_model, dropout=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = self.norm(x)
        out, _ = self.gru(x)
        return r + self.dropout(out)

class VGT_8L_Engine(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([VGT_ResidualBlock(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h0 = self.embedding(x)
        # 捕捉 Layer 0 的输出作为逻辑流形的起点 [cite: 35]
        h_layer0 = self.layers[0](h0)
        
        h = h_layer0
        for layer in self.layers[1:]:
            h = layer(h)
            
        h = self.final_norm(h)
        logits = self.fc(h)
        return logits, h, h_layer0 # 返回 Layer 0 用于几何约束

# ==========================================
# 2. VGT-Pro 核心损失函数 (真正对齐论文)
# ==========================================
def vgt_pro_loss(model, h_layer0, vocab, alpha, lambda_collapse=1.0):
    """
    对齐论文 PCM 理论的改进版损失函数：
    1. 分段 L2 坍缩 (Collapse) 
    2. 边界步长 Drop (4->5) [cite: 7, 68]
    3. 分段平行度 (Piecewise GPA: 0-4线 & 5-9线) 
    """
    device = h_layer0.device
    
    # --- A. 隐藏层 L2 坍缩约束 (温和 L2 压力) ---
    # 论文 4.2 提到 mild L2 pressure 以防止漂移但不消除曲率 
    collapse_loss = h_layer0.norm(p=2, dim=-1).mean() * lambda_collapse

    # --- 获取数字 Embedding 经过 Layer0 的表示 ---
    digits = [vocab.get(str(i), vocab.get("[UNK]", 0)) for i in range(10)]
    digit_ids = torch.tensor(digits, device=device)
    # 捕获 Layer 0 输出作为几何起点 [cite: 35]
    h_digits = model.layers[0](model.embedding(digit_ids)) 
    
    # 计算相邻数字间的位移向量 (步长向量)
    diffs = h_digits[1:] - h_digits[:-1] # 0->1, 1->2 ... 8->9 (共9个向量)
    steps = torch.norm(diffs, dim=1)     # 步长模长
    
    # --- B. PCM 步长分布约束 (4->5 边界) ---
    # 强制 4->5 步长相对于 3->4 显式下降，编码进位规则 [cite: 7, 68]
    step_34 = steps[3]
    step_45 = steps[4]
    pcm_drop_loss = F.mse_loss(step_45, step_34 * 0.456) # 目标下降 ~54.4% [cite: 68]

    # --- C. 分段平行度约束 (Piecewise GPA) ---
    # 论文要求 Intra-segment consistency：段内方向一致 [cite: 45]
    # 段 A: 0-4 (对应 diffs 的索引 0, 1, 2, 3)
    segA_diffs = diffs[0:4] 
    mean_dir_A = segA_diffs.mean(dim=0, keepdim=True)
    cos_sim_A = F.cosine_similarity(segA_diffs, mean_dir_A, dim=1)
    parallel_loss_A = (1 - cos_sim_A).mean()

    # 段 B: 5-9 (对应 diffs 的索引 5, 6, 7, 8)
    # 解决 5-9 坍缩成点的问题：强制 5-9 也要形成一条平行线 [cite: 43, 68]
    segB_diffs = diffs[5:9] 
    mean_dir_B = segB_diffs.mean(dim=0, keepdim=True)
    cos_sim_B = F.cosine_similarity(segB_diffs, mean_dir_B, dim=1)
    parallel_loss_B = (1 - cos_sim_B).mean()
    
    # 总平行度损失：排除 4->5 边界，允许其方向突变 [cite: 46]
    total_parallel_loss = (parallel_loss_A + parallel_loss_B) / 2

    # --- D. 最小步长保护 (防止 5-9 段 Over-collapse) ---
    # 确保段内数字间有最小物理间距，逻辑才能被区分 [cite: 10, 59]
    min_step = 0.05
    step_guard_loss = torch.relu(min_step - steps).mean()

    # 权重对齐论文实验配置 [cite: 55]
    return alpha * (
        0.1 * collapse_loss + 
        5.0 * pcm_drop_loss + 
        2.0 * total_parallel_loss + 
        1.0 * step_guard_loss
    )

# ==========================================
# 3. 训练主逻辑
# ==========================================
def run_vgt_pro_training():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR, BS, EPOCHS, SEQ = 1e-4, 32, 50, 256
    
    # 加载词表
    vocab = json.load(open("vocab.json", "r", encoding="utf-8"))
    model = VGT_8L_Engine(len(vocab)).to(DEVICE)
    
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scaler = torch.amp.GradScaler()

    for ep in range(1, EPOCHS + 1):
        # from train import StreamDataset # 沿用之前的 dataset 类
        dataset = StreamDataset("train_encyclopedia.json", vocab, SEQ)
        loader = DataLoader(dataset, batch_size=BS)
        
        # 动态 Alpha：模拟从统计学习到规则坍缩的过程 [cite: 24]
        alpha = 10.0 if ep < 5 else 50.0 
        lambda_collapse = 1.0 if ep < 10 else 0.5  # 后期mild L2
        for step, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            with torch.amp.autocast("cuda"):
                logits, h_last, h_layer0 = model(x)
                
                # 基础交叉熵损失 [cite: 24]
                ce_loss = F.cross_entropy(logits.view(-1, len(vocab)), y.view(-1))
                
                # VGT-Pro 几何损失
                geom_loss = vgt_pro_loss(model, h_layer0, vocab, alpha,lambda_collapse)
                
                total_loss = ce_loss + geom_loss

            opt.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if step % 50 == 0:
                print(f"Ep {ep} | Step {step} | CE: {ce_loss:.4f} | VGT-Pro: {geom_loss:.4f}")

        torch.save(model.state_dict(), f"vgt_pro_final_ep{ep}.pth")

if __name__ == "__main__":
    run_vgt_pro_training()