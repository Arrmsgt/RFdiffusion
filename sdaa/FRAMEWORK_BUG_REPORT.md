# SDAA 框架 bug 报告：nn.Embedding 三维大输入读未初始化显存

> 本报告记录在 **RFdiffusion** 迁移中再次确认的 Torch-SDAA 算子 bug，
> 与 RoseTTAFold2-PPI 迁移中发现的「框架 bug #2」**完全一致**，
> 两份报告互相印证，强烈建议提交框架研发修复。

## 一、bug 现象

RFdiffusion 的**对称设计**和 **binder/PPI 设计**在 SDAA 上崩溃：

```
numpy.linalg.LinAlgError: SVD did not converge
```

- CUDA（A100）：对称/binder/无条件/motif/partial **全部正常**
- SDAA：无条件/motif/partial 正常，**对称/binder 崩溃**
- CPU：**全部正常**

## 二、定位链路（5 步铁证）

```
① 对称/binder 报 SVD 不收敛
   ↓ scipy Rotation.from_matrix 对含 NaN 的矩阵做 SVD
② R_0 旋转矩阵含 NaN（诊断：R_0 NaN=True, R_t NaN=False）
   ↓ R_0 由模型预测坐标 px0 三点构架
③ px0 含 NaN（SDAA 模型 forward 输出）
   ↓ forward hook 逐层检测
④ 首个 NaN 出现在 latent_emb.pos.emb（PositionalEncoding2D 的 nn.Embedding 查表）
   ↓ 诊断：emb.weight 无 NaN、ib 索引全合法(0-64)、seqsep 无 NaN
⑤ emb(ib) 输出全 NaN → 算子本身 bug
```

## 三、根因

**SDAA 的 `nn.Embedding` 对三维大索引输入读未初始化显存，产生垃圾值/NaN**（gather/index_select 底层 bug）。

关键证据：
- emb.weight 无 NaN（正常权重）
- ib 索引合法（0-64，nbin=65）
- seqsep 无 NaN
- 但查表输出全 NaN

## 四、触发条件

**输入长度 L 决定 pair 特征大小，大输入触发**：

| 输入形状 | 元素数 | 现象 |
|---------|:---:|------|
| [1, 50, 50] | 2,500 | ✅ 正常（无条件生成 L=50）|
| [1, 79, 79] | 6,241 | ✅ 正常（partial L=79）|
| **[1, 240, 240]** | **57,600** | ❌ NaN（对称设计 L=240）|
| **[1, 250, 250]** | **62,500** | ❌ NaN（binder L=250）|

最小复现：`nn.Embedding(65,128)` 对 `(1,200,200)` 随机索引**偶发** 50 个 NaN（其他大小正常），
说明 bug 是**偶发、与内存状态相关**的（读未初始化显存，内容随机）。

## 五、数据类型确认

- 模型 float32（`emb.dtype=torch.float32`），**不是**混合精度/bf16 问题
- 代码 `eps=1e-8` 比 float32 机器精度(1.19e-7)小 11.9 倍，是**次要隐患**（非本次根因）

## 六、workaround（模型侧，原代码注释保留）

```python
# rfdiffusion/Embeddings.py  PositionalEncoding2D.forward
# [SDAA workaround] nn.Embedding 三维大输入读未初始化显存产生 NaN
# 原实现（框架 bug 修好后恢复）：
# emb = self.emb(ib) #(B, L, L, d_model)
emb = torch.nn.functional.one_hot(ib, num_classes=self.nbin).float() @ self.emb.weight
```

`one_hot + matmul` 与原 `Embedding` 查表**数值完全等价**（vocab 仅 65，绕开大输入 gather bug）。
验证：对称设计（C3/240 残基）从「第 1-2 步 SVD 崩」到「完整跑通 7.32 分钟」。

## 七、影响范围

任何在 SDAA 上对 `nn.Embedding` 做**三维大索引输入**（元素数 ≥ 数万）的代码都会踩，
不仅限于 RFdiffusion/RoseTTAFold2-PPI——任何蛋白折叠/设计模型（pair 特征是 [L,L] 索引）都可能触发。

## 八、与 RoseTTAFold2-PPI 的关联

RoseTTAFold2-PPI 的「框架 bug #2」在 `PositionalEncoding2D.self.emb` 用 [1,615,615] 输入时
读未初始化显存。RFdiffusion 在 [1,240,240] 输入时**再次复现同一 bug**，确认是系统性框架问题，
建议框架研发优先修复 `nn.Embedding`（gather/index_select）对大输入的内存正确性。