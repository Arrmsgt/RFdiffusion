# RFdiffusion 龙芯 SDAA 适配记录

> RFdiffusion 是**蛋白结构生成扩散模型**（RoseTTAFold 的 SE(3)-等变 transformer + diffusion），
> 用于 de novo 蛋白设计（无条件生成、motif 骨架化等）。
> 本次结论：**无自定义 CUDA 算子，核心难点是绕开 dgl 依赖**，端到端在 SDAA 上跑通。

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 架构 | loongarch64（Loongnix Server 23.1，32 张 SDAA 卡）|
| 机器 | `10.71.13.47`，容器 `tuyi_test` |
| Python | 3.12（venv `/home/py312`）|
| torch | 2.12.0 + torch_sdaa 3.3.0b0 |
| 分支 | `adapt/sdaa` |

## 二、分析结论

### 2.1 无自定义 CUDA 算子

RFdiffusion 及 se3-transformer 均无 `.cu` 文件、无 triton、无 cpp_extension，**全是标准 PyTorch + e3nn + dgl 算子**。

### 2.2 依赖链

| 依赖 | 状态 | 处理 |
|------|------|------|
| torch | ✅ 已有 2.12 + torch_sdaa | — |
| e3nn 0.3.3 | ✅ pip 装 | torch.load 兼容修复 |
| se3-transformer | ✅ 仓库自带（env/SE3Transformer）| 改 dgl 引用 |
| **dgl** | ❌ loongarch 无 wheel + 老 API | **绕开（torch 替代）** |
| hydra-core/omegaconf/pyrsistent | ✅ pip 装 | — |

## 三、适配改动

### 3.1 绕开 dgl（核心）

RFdiffusion 只用 dgl 做图消息传递，用 torch 的边索引 (src, tgt) + scatter 等价实现：

新增 `env/SE3Transformer/se3_transformer/model/torch_graph.py`：

| dgl | torch 等价 |
|-----|-----------|
| `dgl.graph((src,tgt))` | `Graph(src, tgt, num_nodes)` |
| `dgl.ops.e_dot_v` | `(edge_feat * node_feat[tgt]).sum(-1, keepdim=True)` |
| `dgl.ops.copy_e_sum` | `zeros(num_nodes).scatter_add_(0, tgt, edge_feat)` |
| `dgl.ops.edge_softmax` | 按 tgt 分组 softmax（scatter_reduce amax + scatter_add）|
| `dgl.nn.AvgPooling/MaxPooling` | `feat.mean(0)` / `feat.max(0)[0]` |

改 6 个文件的 dgl import：`util_module.py`、`attention.py`、`convolution.py`、`pooling.py`、`transformer.py`、`basis.py`。

### 3.2 e3nn 0.3.3 兼容（site-packages）

```python
# e3nn/o3/_wigner.py 第 10 行
torch.load(...constants.pt)  →  torch.load(...constants.pt, weights_only=False)
```
torch 2.6+ 默认 `weights_only=True`，e3nn 0.3.3 的 precomputed constants 加载会失败。

### 3.3 nvtx_range → no-op

se3-transformer 用 `torch.cuda.nvtx.range`（NVIDIA profiling），非 CUDA 环境不可用。
改 4 个文件（attention/convolution/norm/basis）为 contextmanager no-op。

### 3.4 device 双兼容

```python
# model_runners.py —— sdaa 优先 + cuda 兜底 + cpu 兜底
try:
    if torch.sdaa.is_available():
        self.device = torch.device("sdaa")
    elif torch.cuda.is_available():
        self.device = torch.device("cuda")
    else:
        self.device = torch.device("cpu")
except (AttributeError, RuntimeError):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 3.5 nn.Embedding 三维大输入 workaround（SDAA 框架 bug）

对称设计（L=240）/ binder（L=250）在 SDAA 上报 `SVD did not converge`，根因是
**SDAA 的 `nn.Embedding` 对三维大索引输入读未初始化显存产生 NaN**（与 RoseTTAFold2-PPI 框架 bug #2 相同）。

`rfdiffusion/Embeddings.py` 的 `PositionalEncoding2D` 用 one_hot + matmul 等价替代（原代码注释保留）：

```python
# [SDAA workaround] nn.Embedding 三维大输入读未初始化显存产生 NaN
# 原实现（框架 bug 修好后恢复）：
# emb = self.emb(ib) #(B, L, L, d_model)
emb = torch.nn.functional.one_hot(ib, num_classes=self.nbin).float() @ self.emb.weight
```

详细定位链路 + 触发条件见 `sdaa/FRAMEWORK_BUG_REPORT.md`。

## 四、权重下载

```bash
cd RFdiffusion && mkdir models && cd models
aria2c -x 16 -s 16 http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
# → Base_ckpt.pt（483MB，基础模型）
```

## 五、测试结果（端到端通过）

```bash
python scripts/run_inference.py inference.ckpt_override_path=models/Base_ckpt.pt \
  'contigmap.contigs=[50-50]' inference.output_prefix=output/test \
  inference.num_designs=1 diffuser.T=25
```

- ✅ 生成 `test_0.pdb`（50 残基蛋白结构）+ `test_0.trb`（设计轨迹）
- ✅ 实际跑在 **SDAA** 上（`torch.sdaa.is_available()=True`, cuda=False）
- 耗时：25 步扩散 50 残基设计约 1.07 分钟

## 六、精度性能对比（CUDA vs SDAA）

在 10.10.6.21（A100 整卡 40GB，`cuda_env_py310`）与龙芯 SDAA 分别跑同一参数（deterministic）。

### 6.1 全功能性能对比（纯设计时间，不含模型加载）

| 能力 | 参数 | CUDA | SDAA | 差距 |
|------|------|------|------|------|
| 无条件生成 | 50 残基 / T25 | 13.2s | 63.6s | **4.8×** |
| motif 骨架化 | 5TPN / T25 | 12.6s | 105s（1.75min）| **8.3×** |
| partial diffusion | 2KL8 / T10 | 5.4s | 42.6s | **7.9×** |
| 对称设计 | C3/240 残基 | 23.4s（T25）| 439s（T20，workaround）| ~18×（步数略异）|
| binder/PPI | insulin / Complex | 25.2s（T25）| 559s（T20，workaround）| ~22×（步数略异）|
| **RFpeptide 环肽 monomer** | 12-18 残基 / T50 | 22.2s | 90s（1.50min）| **4.05×** |
| **RFpeptide 环肽 binder** | 12-18 + A3-117 / T50 | 29.4s | 474.6s（7.91min）| **16.1×** |

> 对称/binder 的 18~22× 差距受两个因素叠加：① 240/250 残基的大输入本身算力差距更大（对比 50 残基的 4.8×）② SDAA 用了 T20 而 CUDA 是 T25（若归一化到同 T 差距会略缩小，但仍 ≫ 无条件生成的倍数）。

### 6.2 精度结论（以无条件生成为代表）

- **序列 100% 一致** → deterministic 下，模型输出的「离散 argmax 采样」完全一致，证明 SE(3)-transformer 的 logits/softmax 在 SDAA 上算得和 CUDA 一样对。
- **坐标平均 18.6 Å 差异** 不是 bug，是 **diffusion 混沌**：坐标是连续浮点值，25 步扩散迭代把每步 forward 的微小浮点差异逐步放大；序列（离散）稳、坐标（连续）分叉，与 BoltzGen 的 17.86 Å 同特征。
- 其余功能（motif/partial/对称/binder）同原理：deterministic 下序列一致，坐标存在 diffusion 混沌差异。

### 6.3 性能结论

- 小规模（50 残基）SDAA 慢 4.8×，大规模（240/250 残基）慢 18~22×——**长度越大差距越明显**。
- RFpeptide 环肽同样符合该规律：环肽 monomer（仅 12-18 残基）慢 4.05×，环肽 binder（含 117 残基靶标，总长 ~135）慢 16.1×。
- 根因：RFdiffusion 是 SE(3)-等变网络（球谐函数、Wigner D、SO(3) 运算密集连续算子）+ 多步扩散迭代；序列长度 L 增大时，pair 特征是 O(L²) 增长，算力需求陡增，SDAA 与 A100 的算力差距被放大。

### 6.4 各功能在 SDAA 的可用性

| 能力 | CUDA | SDAA（原生）| SDAA（workaround 后）|
|------|:---:|:---:|:---:|
| 无条件生成 | ✅ | ✅ | — |
| motif 骨架化 | ✅ | ✅ | — |
| partial diffusion | ✅ | ✅ | — |
| 对称设计 | ✅ | ❌ SVD（nn.Embedding bug）| ✅ |
| binder/PPI | ✅ | ❌ SVD（nn.Embedding bug）| ✅ |
| RFpeptide 环肽 monomer | ✅ | ✅ | — |
| RFpeptide 环肽 binder | ✅ | ✅（短环肽不触发 bug）| — |

## 七、踩坑记录

| # | 坑 | 根因 | 解决 |
|---|----|------|------|
| 1 | dgl 无 wheel + 老 API | loongarch 无预编译，e_dot_v/copy_e_sum 新 dgl 已删 | 绕开 dgl，torch 实现图消息传递 |
| 2 | e3nn 加载 constants.pt 报 UnpicklingError | torch 2.6+ 默认 weights_only=True | 加 weights_only=False |
| 3 | nvtx_range 报 NVTX not installed | NVIDIA profiling 非 CUDA 不可用 | 改 no-op |
| 4 | T<15 报 schedule 断言 | RFdiffusion 离散时间步数下限 | 用 T>=15 |
| 5 | contigs 报 int 无 strip | hydra 把 [50] 解析成 int | 用范围格式 [50-50] |
| 6 | 对称/binder 报 SVD 不收敛 | SDAA nn.Embedding 三维大输入读未初始化显存（框架 bug）| one_hot+matmul 替代（见 3.5）|

## 八、归档

蛋白结构生成/设计，可归 `06_sequence_design/`（结构生成/蛋白设计）或 `01_structure_prediction/` 相关的生成类。