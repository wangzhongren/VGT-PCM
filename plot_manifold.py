import torch
import torch.nn.functional as F
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import json
import os
# 导入模型定义 (假设与训练脚本在同一文件或已导入)
# from your_module import VGT_8L_Engine 

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

def load_vgt_model(checkpoint_path, vocab_path, d_model=512):
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    inv_vocab = {v: k for k, v in vocab.items()}
    
    model = VGT_8L_Engine(len(vocab), d_model=d_model)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model, vocab, inv_vocab

def generate_geometric_vgt(model, vocab, inv_vocab, prompt, max_len=32, device="cpu"):
    model.to(device)
    model.eval()
    
    input_ids = [vocab.get(c, vocab["[UNK]"]) for c in prompt]
    input_tensor = torch.LongTensor([input_ids]).to(device)
    
    generated_text = prompt
    
    # 记录已生成的字符，用于惩罚重复（防止陷入“右心室”循环）
    generated_ids = set(input_ids) 

    with torch.no_grad():
        for _ in range(max_len):
            # 1. 提取最后一层的几何表示 h_last
            logits, h_last, _ = model(input_tensor)
            
            # 2. 纯几何投影：手动执行 W * h，跳过 Bias
            # 这样可以消除统计偏置（Bias）对逻辑方向的干扰
            weight = model.fc.weight  # [vocab_size, d_model]
            last_hidden = h_last[0, -1, :] # [d_model]
            
            # 计算纯粹的余弦相似度或内积投影
            geo_logits = torch.matmul(weight, last_hidden) 
            
            # 3. 简单的重复惩罚 (Repetition Penalty)
            # 如果某个字符已经出现过，稍微降低它的分数，强迫流形向下一步跳转
            for gid in generated_ids:
                geo_logits[gid] -= 2.0 

            # 4. Argmax 取纯几何落点
            next_token_id = torch.argmax(geo_logits).item()
            next_char = inv_vocab.get(next_token_id, "")
            
            # 终止符判断
            if next_char == "[EOS]" or next_char == " ": 
                break 
            
            # 更新状态
            generated_text += next_char
            generated_ids.add(next_token_id)
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)
            
    print(f"[Geometric Generated]: {generated_text}")
    return generated_text


def run_scientific_validation(model, vocab, inv_vocab):
    # 自动获取模型所在的设备 (cuda 或 cpu)
    device = next(model.parameters()).device 
    model.eval()
    
    print("\n" + "="*50)
    print("      VGT 逻辑流形实证报告 (Scientific Report)")
    print("="*50)

    # --- 验证 1: 几何拓扑 (0-9 数字链) ---
    digits = [str(i) for i in range(10)]
    # 修复：确保输入张量在正确的设备上
    digit_ids = torch.tensor([vocab[d] for d in digits]).to(device) 
    
    with torch.no_grad():
        # 提取 Layer 0 的表示
        h_emb = model.embedding(digit_ids)
        h_digits = model.layers[0](h_emb)
        
        # 计算欧氏距离步长
        diffs = h_digits[1:] - h_digits[:-1]
        steps = torch.norm(diffs, dim=1)
        
        step_34 = steps[3].item()
        step_45 = steps[4].item()
        drop_rate = (step_34 - step_45) / step_34 if step_34 > 0 else 0
        
        print(f"[几何验证] 步长 3->4: {step_34:.4f}")
        print(f"[几何验证] 步长 4->5: {step_45:.4f}")
        print(f"[几何验证] 4->5 坍缩率: {drop_rate*100:.2f}% (预期目标: ~54.4%)")
        
        # 计算平行度 (Cosine Similarity)
        cos_sim = F.cosine_similarity(diffs[0:3], diffs[1:4]).mean()
        print(f"[几何验证] 段内平行度 (GPA): {cos_sim:.4f} (越接近 1 说明逻辑越线性)")

        # --- 验证 2: 加法流形类比 (Analogy Test) ---
        print("\n[类比验证] 正在测试跨域加法位移...")
        
        def get_vec(word):
            if word not in vocab: return None
            idx = torch.tensor([vocab[word]]).to(device) # 修复设备一致性
            return model.layers[0](model.embedding(idx))

        # 验证公式: 巴黎 + (中国 - 北京) ≈ 法国
        pairs = [("北京", "中国", "巴黎", "法国")]
        for w1, w2, w3, w_target in pairs:
            v1 = get_vec(w1)
            v2 = get_vec(w2)
            v3 = get_vec(w3)
            vt = get_vec(w_target)
            
            if v1 is not None and vt is not None:
                # 执行向量加法运算
                v_logic = v3 + (v2 - v1)
                sim = F.cosine_similarity(v_logic, vt)
                print(f"  测试结果: {w1}:{w2} = {w3}:? ")
                print(f"  -> 与目标词 '{w_target}' 的几何相似度: {sim.item():.4f}")
                if sim.item() > 0.6:
                    print("  [结论] 实证成功：逻辑被编码为空间平移向量。")

    # --- 验证 3: PCA 降维可视化 ---
    print("\n[绘图] 正在生成流形拓扑图...")
    data = h_digits.detach().cpu().numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-muted')
    
    # 绘制分段流形
    plt.plot(coords[:5, 0], coords[:5, 1], 'g--', alpha=0.6, label='0-4 Segment')
    plt.plot(coords[4:6, 0], coords[4:6, 1], 'r-', linewidth=3, label='4-5 Logic Boundary')
    plt.plot(coords[5:, 0], coords[5:, 1], 'b--', alpha=0.6, label='5-9 Segment')
    plt.scatter(coords[:, 0], coords[:, 1], c='black', zorder=3)

    for i, d in enumerate(digits):
        plt.annotate(d, (coords[i, 0], coords[i, 1]), xytext=(5, 5), textcoords='offset points')
    
    plt.title("VGT Piecewise Constrained Manifold Visualization")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.show()

def test_mathematical_geometry(model, vocab):
    """验证数字 0-9 在隐藏层是否满足 VGT 几何约束"""
    print("\n--- 几何拓扑验证: 0-9 数字链 ---")
    device = next(model.parameters()).device 
    model.eval()
    
    print("\n" + "="*40)
    print("      VGT 逻辑流形实证报告")
    print("="*40)
    digits = [str(i) for i in range(10)]
    digit_ids = torch.tensor([vocab[d] for d in digits]).to(device)
    
    with torch.no_grad():
        # 提取 Layer 0 的嵌入表示
        h_digits = model.layers[0](model.embedding(digit_ids))
        
        # 计算欧氏距离步长
        diffs = h_digits[1:] - h_digits[:-1]
        steps = torch.norm(diffs, dim=1)
        
        step_34 = steps[3].item()
        step_45 = steps[4].item()
        drop_ratio = (step_34 - step_45) / step_34 if step_34 > 0 else 0
        
        print(f"步长 3->4: {step_34:.4f}")
        print(f"步长 4->5: {step_45:.4f}")
        print(f"4->5 坍缩率: {drop_ratio*100:.2f}% (目标: ~54.4%)")
        
        # 计算平行度 (Cosine Similarity)
        cos_sim = F.cosine_similarity(diffs[0:3], diffs[1:4]).mean()
        print(f"段内平行度 (GPA): {cos_sim:.4f} (越接近 1 说明逻辑越线性)")

def run_tests(model_path, vocab_path):
    model, vocab, inv_vocab = load_vgt_model(model_path, vocab_path)
    
    # 1. 几何对齐测试
    test_mathematical_geometry(model, vocab)
    
    # 2. 数学加法逻辑测试
    # 考察模型在经过几何约束后，是否能通过“向量平移”完成简单加法
    math_prompts = [
        "1+1=",
        "4+1=",
        "3+2=",
        "8+1="
    ]
    math_prompts = [
    # --- 区间 A: 0-4 线性段验证 (目前 PCA 图中成线部分) ---
    "1+1=", "1+2=", "1+3=", "2+1=", "2+2=", "3+1=", "0+4=", 
    
    # --- 边界跨越: 4->5 逻辑断点 (测试规则边界跳转) ---
    "4+1=", "3+2=", "2+3=", "1+4=", "5+0=",
    
    # --- 区间 B: 5-9 坍缩段验证 (目前 PCA 图中聚成一点的部分) ---
    "5+1=", "6+1=", "7+1=", "8+1=", "5+2=", "5+3=", "5+4=", "4+4="
]

    # 循环执行测试
  
    for p in math_prompts:
        generate_geometric_vgt(model, vocab, inv_vocab, p, max_len=5)

    # 3. 医学逻辑测试
    # 考察 VGT 是否捕捉到了医学知识中的线性因果关系
    med_prompts = [
        "心脏位于纵隔的",
        "阿司匹林的作用机制是抑制",
        "多饮、多尿、多食、体重减轻是典型的",
        "心率过快称为",
        "今天天气不错",
        "今天天气不错，阿司匹林的作用是"
    ]
    print("\n--- 任务测试: 医学逻辑 ---")
    for p in med_prompts:
        generate_geometric_vgt(model, vocab, inv_vocab, p, max_len=50)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_scientific_validation(model, vocab, inv_vocab):
    model.eval()
    print("\n" + "="*40)
    print("      VGT 逻辑流形实证报告")
    print("="*40)

    # --- 验证 1: 几何拓扑 ---
    digits = [str(i) for i in range(10)]
    digit_ids = torch.tensor([vocab[d] for d in digits])
    with torch.no_grad():
        h_digits = model.layers[0](model.embedding(digit_ids))
        diffs = h_digits[1:] - h_digits[:-1]
        steps = torch.norm(diffs, dim=1)
        
        drop_rate = (steps[3] - steps[4]) / steps[3]
        print(f"[几何] 4->5 坍缩率: {drop_rate.item()*100:.2f}% (目标: ~54.4%) [cite: 68]")
        
        # --- 验证 2: 逻辑类比 (加法流形实证) ---
        def get_vec(word):
            if word not in vocab: return None
            idx = torch.tensor([vocab[word]])
            return model.layers[0](model.embedding(idx))

        print("\n[类比] 正在测试加法流形位移...")
        pairs = [("北京", "中国", "巴黎", "法国")]
        for w1, w2, w3, w_target in pairs:
            v1, v2, v3, vt = get_vec(w1), get_vec(w2), get_vec(w3), get_vec(w_target)
            if v1 is not None and vt is not None:
                # 核心公式: 巴黎 + (中国 - 北京)
                v_logic = v3 + (v2 - v1)
                sim = F.cosine_similarity(v_logic, vt)
                print(f"  {w1}:{w2} = {w3}:? -> 目标({w_target}) 几何相似度: {sim.item():.4f}")

    # --- 验证 3: PCA 可视化 ---
    data = h_digits.detach().cpu().numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    plt.plot(coords[:5, 0], coords[:5, 1], 'g--', label='0-4 Segment [cite: 68]')
    plt.plot(coords[4:6, 0], coords[4:6, 1], 'r-', linewidth=3, label='Boundary [cite: 7]')
    plt.plot(coords[5:, 0], coords[5:, 1], 'b--', label='5-9 Segment')
    for i, d in enumerate(digits): plt.annotate(d, (coords[i,0], coords[i,1]))
    plt.title("Manifold Visualization: Logic as Geometric Invariant [cite: 39]")
    plt.legend()
    plt.show()
# 确保导入你定义的模型类
# from your_model_file import VGT_8L_Engine

def visualize_digit_manifold(checkpoint_path, vocab_path):
    # 1. 加载词表
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # 2. 初始化模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGT_8L_Engine(len(vocab))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. 准备数字数据 (0-9)
    digits = [str(i) for i in range(10)]
    try:
        digit_ids = torch.tensor([vocab[d] for d in digits]).to(device)
    except KeyError as e:
        print(f"错误：词表中缺少数字 token {e}")
        return

    # 4. 提取 Layer 0 的几何表示
    with torch.no_grad():
        # 模拟 forward 流程中提取 h_layer0 的逻辑
        h_emb = model.embedding(digit_ids)
        # 这里必须调用 model.layers[0]，对应训练时的几何约束点
        h_layer0 = model.layers[0](h_emb) 
        
        # 转换为 numpy (Shape: [10, d_model])
        data = h_layer0.cpu().numpy()

    # 5. 执行 PCA 降维
    pca = PCA(n_components=2)
    coords = pca.fit_transform(data)
    
    # 计算解释方差比 (PC1 + PC2 占比越高，说明流形越扁平/线性)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100

    # 6. 绘图
    plt.figure(figsize=(10, 8), dpi=120)
    plt.style.use('seaborn-v0_8-muted')
    
    # 绘制坐标点
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, zorder=2)
    
    # 绘制连线 (展现逻辑轨迹)
    plt.plot(coords[:5, 0], coords[:5, 1], 'g--', alpha=0.6, label='0-4 Linear Segment')
    plt.plot(coords[4:6, 0], coords[4:6, 1], 'r-', linewidth=3, label='4-5 Collapse Edge')
    plt.plot(coords[5:, 0], coords[5:, 1], 'b--', alpha=0.6, label='5-9 Recovery Segment')

    # 为每个点添加标签
    for i, txt in enumerate(digits):
        plt.annotate(txt, (coords[i, 0], coords[i, 1]), 
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=12, fontweight='bold')

    plt.title(f"VGT-Pro Digit Logic Manifold\n(PC1+PC2 Explained: {explained_var:.2f}%)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # 保存并展示
    plt.savefig("digit_manifold_vgt.png")
    print("可视化完成！结果已保存至 digit_manifold_vgt.png")
    plt.show()

def run2():
    # 设定路径
    CHECKPOINT_PATH = "vgt_pro_final_ep14.pth"  # 你的模型权重文件
    VOCAB_PATH = "vocab.json"                   # 你的词表文件

    # 1. 加载词表
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    # 2. 初始化模型架构 (确保参数与训练时一致)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGT_8L_Engine(vocab_size=len(vocab), d_model=512)

    # 3. 加载训练好的权重
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"成功加载权重: {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print("错误：未找到权重文件，请确认路径是否正确。")

    # 4. 执行实证验证
    # 这个函数会依次执行：几何拓扑计算、逻辑类比测试、PCA 降维可视化
    run_scientific_validation(model, vocab, inv_vocab)


# ==========================================
# 2. 字符级医学语义分布分析
# ==========================================
def analyze_char_medical_manifold(model_path, vocab_path, d_model=512):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGT_8L_Engine(len(vocab), d_model=d_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # --- A. 准备数字基准 (0-9) ---
    digits = [str(i) for i in range(10)]
    digit_ids = torch.tensor([[vocab[d] for d in digits]], dtype=torch.long).to(device)

    # --- B. 准备医学关键词（自动拆分为字符） ---
    med_terms = ["心脏", "阿司匹林", "抑制", "血小板", "多尿", "2.5", "1000"]
    
    med_char_vecs = []
    med_labels = []

    with torch.no_grad():
        # 1. 提取数字流形
        _, _, h_digits_raw = model(digit_ids)
        digit_vecs = h_digits_raw.squeeze(0).cpu().numpy() # [10, d_model]

        # 2. 提取医学词汇的字符几何中心
        for term in med_terms:
            # 将词拆分为单字符，并过滤掉不在词表里的字符
            chars = [c for c in term if c in vocab]
            if not chars: continue
            
            char_ids = torch.tensor([[vocab[c] for c in chars]], dtype=torch.long).to(device)
            # 通过 Layer 0 提取这些字符的隐藏状态
            _, _, h_chars = model(char_ids) 
            
            # 取该词所有字符向量的平均值，代表该医学概念在流形上的位置
            term_vec = h_chars.mean(dim=1).squeeze(0).cpu().numpy()
            med_char_vecs.append(term_vec)
            med_labels.append(term_vec) # 记录标签
            print(f"[处理] 医学词 '{term}' 拆解为 {len(chars)} 个字符进行向量合成")

    # --- C. PCA 降维与 GIT 诊断 ---
    med_char_vecs = np.array(med_char_vecs)
    all_data = np.concatenate([digit_vecs, med_char_vecs], axis=0)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_data)
    
    digit_coords = coords[:10]
    med_coords = coords[10:]

    # --- D. 绘图 ---
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制数字骨架
    plt.plot(digit_coords[:5, 0], digit_coords[:5, 1], 'g--', alpha=0.4, label='0-4 逻辑段')
    plt.plot(digit_coords[4:6, 0], digit_coords[4:6, 1], 'r-', linewidth=3, label='4-5 坍缩边界')
    plt.plot(digit_coords[5:, 0], digit_coords[5:, 1], 'b--', alpha=0.4, label='5-9 逻辑段')
    plt.scatter(digit_coords[:, 0], digit_coords[:, 1], c='black', marker='x')

    # 绘制医学概念点
    plt.scatter(med_coords[:, 0], med_coords[:, 1], c='red', s=100, edgecolors='white', label='医学语义结晶')

    print("\n--- GIT (几何身份理论) 捕获诊断 ---")
    for i, label in enumerate(med_terms):
        if i >= len(med_coords): break
        plt.annotate(label, (med_coords[i, 0], med_coords[i, 1]), xytext=(5, 5), textcoords='offset points')
        
        # 诊断：哪个数字被该医学词“捕获”了坐标
        dists = np.linalg.norm(digit_coords - med_coords[i], axis=1)
        nearest = np.argmin(dists)
        print(f"医学概念 '{label: <5}' -> 几何捕获数字 '{nearest}' (距离: {dists[nearest]:.4f})")

    plt.title("字符级 VGT 医学-逻辑流形分布图")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # run_tests("vgt_pro_final_ep50.pth", "vocab.json");
    # 请确保路径下有对应的文件
    # visualize_digit_manifold("vgt_pro_final_ep17.pth", "vocab.json")
                  # 你的词表文件
    analyze_char_medical_manifold("vgt_pro_final_ep50.pth", "vocab.json")
    # 1. 加载词表
    # with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    #     vocab = json.load(f)
    # inv_vocab = {v: k for k, v in vocab.items()}

    # # 2. 初始化模型架构 (确保参数与训练时一致)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = VGT_8L_Engine(vocab_size=len(vocab), d_model=512)
    # # run_tests("vgt_pro_final_ep27.pth", "vocab.json");
    # run2();
    # analyze_medical_distribution(model, vocab, inv_vocab)

    # print("请确认已训练模型并指定正确的路径。")