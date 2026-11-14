# LatentChem pKa — 诊断日志（修 token ID + 切文本捷径）

> 记录时间：2026-11-12
> 状态：token ID 已修复 ✓ / 文本捷径已切断 ✓ / 重训进行中

## 一、问题背景

Fig 3（因果干预实验）失败：改变 latent 向量后，PkaHead 读数大幅移动（Δhead=+2.4），但生成的文本完全不变（Δtext=0.00）。这表明 latent 没有因果驱动 LM 主路径的文本输出。

进一步诊断发现**两个独立的架构缺陷**：

---

## 二、根因 A：Token ID 错位

### 现象
- dataloader 的 `LATENT_ID=151669`，model 的 `latent_id=151671`
- 数据集写入 6 个 `<latent>` 位置（ID=151669），但模型只认 `model.latent_id=151671`
- 结果：6 个 latent slot 中只有 1 个（恰好是 dataset 的 `<end_latent>` 位置）被模型识别为 latent

### 根因
`dataloader.py` 和 `model_stage3.py` 各自调用 `tokenizer.add_tokens()`，但 token 列表**顺序不同**：

| 文件 | add_tokens 顺序 |
|------|----------------|
| dataloader（旧） | `[latent, start_latent, end_latent, mol_start, mol_end]` |
| model（旧） | `[mol_start, mol_end, latent, start_latent, end_latent, bio_latent, start_bio_latent, end_bio_latent]` |

`add_tokens` 按追加顺序分配 ID，导致同一字符串在数据和模型中映射到不同 ID。

### 修复
在 `config.py` 定义单一真相源 `EXTRA_TOKENS`，dataloader 和 model 都从 config 导入：

```python
# config.py
EXTRA_TOKENS = [
    "<mol_start>", "<mol_end>", "<latent>", "<start_latent>",
    "<end_latent>", "<bio_latent>", "<start_bio_latent>", "<end_bio_latent>",
]
```

修复后对齐验证：
```
dl   LATENT/START/END = 151671 151672 151673
model latent/start/end= 151671 151672 151673
✓ token IDs aligned
```

### 验证（probe_tokenfix.json）
- **K=6 latent slots** 全部被模型识别（之前 K=1）
- 清零所有 6 个 latent 嵌入：text changed 6/12，pred changed 6/12（变化幅度小）

**结论**：token ID 修复恢复了 latent 通路（K=6），但仅修 ID 不够——文本捷径仍主导输出。

---

## 三、根因 B：文本捷径（Text Shortcut）

### 现象
模态冲突实验（n=24）：

| 条件 | MAE vs source GT | MAE vs donor GT |
|------|-----------------|----------------|
| base | 1.17 | 4.06 |
| text→donor, mol→source | 4.08 | **1.20** |
| mol→donor, text→source | **1.17** | 4.07 |

**答案 100% 由 query 中的 SMILES 字符串决定**，smi-ted 分支和 latent slot 对生成无影响。

### 根因
query 模板为 `"Please estimate the pKa of this molecule: {smiles}."`，SMILES 直接出现在文本中。LM 可以直接从字符串"读"出 pKa，完全绕过 mol 分支和 latent。

### 修复
在 `dataloader.py` 的 `extract_fields` 中，对 pKa 任务移除 query 中的 SMILES：

```python
if benchmark_task == "pka":
    for s in input_smiles:
        query = query.replace(s, "")
    query = re.sub(r'molecule\s*:\s*\.?', 'molecule', query)
```

修复后 query 变为：`"Please estimate the pKa of this molecule"`（无 SMILES）。
分子结构仍通过 `input_smiles` → smi-ted encoder 输入模型。

---

## 四、"旁挂读码器"现象（Side-Mounted Reader）

Fig 3 失败的核心发现：PkaHead 是一个**强读码器**，能从一个向量读出很准的 pKa，但这个向量没有被 LM 主路径使用。

### 原始 Fig 3 数据（fig3_causal.json，修复前）

- `n_latent = 1`（token ID 错位导致 6 个 latent 只有 1 个生效）
- `ref_head MAE = 1.103`（PkaHead 读数很准）

**干预实验**（shift latent 朝目标 pKa，n=24）：

| dt | head_before→head_after | text_before→text_after | text 变化 |
|----|----------------------|----------------------|----------|
| -1.5 | 大幅移向 target（如 -0.55→-3.41, target=-3.33） | 几乎不变（-0.9→-0.9） | 2/24 微变 |
| +1.0 | 大幅移向 target（如 4.04→6.68, target=6.80） | 几乎不变（3.62→3.62） | 5/24 微变 |
| +2.0 | 大幅移向 target（如 4.04→7.76, target=7.80） | 几乎不变（3.62→3.62） | 2/24 微变 |

**Swap 实验**（交换两分子 latent）：text_changed a=0/12, b=0/12 — 完全无影响。

**解读**：h_T 对 PkaHead 的贡献主要来自"这个位置存在"这个事实（attention 聚合效应），而非该位置携带的分子特异性信息。PkaHead 真正依赖的是整个序列的 attention 聚合，不是最后一个 latent slot 的内容。

---

## 五、切断文本捷径后的 B' 基线

| 模型 | 条件 | MAE (test) |
|------|------|-----------|
| B' (旧) | 有 SMILES in query | 1.43 |
| B' (fix) | 无 SMILES in query | **2.363** |

B' 无 latent 通路，只有 pka_head 读最终 hidden state。切断捷径后 MAE 上升 0.93，说明之前的性能约 65% 来自文本捷径。

**待 C 组重训完成后比较**：如果 C (latent) 显著优于 B' (2.36)，说明 latent 推理真正提取了 smi-ted 分支的物理信息。

---

## 六、当前重训状态

| 作业 | Job ID | 状态 |
|------|--------|------|
| B'_fix | - | ✓ 完成 (MAE=2.363) |
| C_fix | 1644316 | 🔄 训练中 (sub-stage 2) |
| A_fix | 1644536 | ⏳ 排队/运行 |
| B_fix | 1644501 | 🔄 运行中 |

---

## 七、后续计划

1. **等 C_fix 训练完成** → 评测 MAE，与 B'_fix (2.363) 比较
2. **重跑 Fig 1/2/3** 使用修正后的 pipeline（K=6 latent slots, no text shortcut）
3. **如果 C > B' 且 latent 干预能改变文本** → 世界模型叙事成立
4. **如果 C ≈ B'** → latent 推理在此架构不成立，转为批判性研究
