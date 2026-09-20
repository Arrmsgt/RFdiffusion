"""torch 实现的轻量图模块，替代 DGL（绕开 dgl 依赖）。

RFdiffusion / se3-transformer 只用 dgl 做图消息传递，这里用 torch 的边索
引 (src, tgt) + scatter/index 操作等价实现，语义与 dgl 0.x 一致：

- dgl.graph((src,tgt))            -> Graph(src, tgt, num_nodes)
- dgl.ops.e_dot_v(graph, e, v)    -> e_dot_v:  边特征 e 与目标节点特征 v 点积
- dgl.ops.copy_e_sum(graph, e)    -> copy_e_sum: 边特征 sum 聚合到目标节点
- dgl.ops.edge_softmax(graph, l)  -> edge_softmax: 按目标节点分组 softmax
- dgl.nn.{Avg,Max}Pooling         -> AvgPooling / MaxPooling: 节点特征池化
"""
import torch
import torch.nn as nn


class Graph:
    """轻量图，用边索引 (src, tgt) 表示。edata 存边特征（如 'rel_pos'）。"""

    def __init__(self, src, tgt, num_nodes, edata=None, ndata=None):
        self.src = src
        self.tgt = tgt
        self.num_nodes = num_nodes
        self.edata = edata if edata is not None else {}
        self.ndata = ndata if ndata is not None else {}

    def to(self, device):
        self.src = self.src.to(device)
        self.tgt = self.tgt.to(device)
        self.edata = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in self.edata.items()}
        self.ndata = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in self.ndata.items()}
        return self

    @property
    def device(self):
        return self.src.device

    def edges(self, *args, **kwargs):
        """返回 (src, dst) 边索引，等价 dgl.DGLGraph.edges()。"""
        return self.src, self.tgt


def e_dot_v(graph, edge_feat, node_feat):
    """边特征 edge_feat 与每条边目标节点特征 node_feat[tgt] 的点积。

    等价 dgl.ops.e_dot_v：对每条边 e=(u,v) 算 dot(edge_feat[e], node_feat[v])。
    """
    return (edge_feat * node_feat[graph.tgt]).sum(dim=-1, keepdim=True)


def copy_e_sum(graph, edge_feat):
    """把边特征 edge_feat 复制到目标节点并 sum 聚合。

    等价 dgl.ops.copy_e_sum：out[v] = sum_{e: dst(e)=v} edge_feat[e]
    """
    out = torch.zeros((graph.num_nodes, *edge_feat.shape[1:]),
                      device=edge_feat.device, dtype=edge_feat.dtype)
    tgt_expanded = graph.tgt.reshape(-1, *([1] * (edge_feat.dim() - 1))).expand_as(edge_feat)
    out.scatter_add_(0, tgt_expanded, edge_feat)
    return out


def edge_softmax(graph, logits):
    """按目标节点分组做 softmax（数值稳定版）。

    等价 dgl.ops.edge_softmax：对每个目标节点 v，把入射边的 logits 归一化。
    """
    tgt = graph.tgt
    shape = logits.shape
    logits_flat = logits.reshape(shape[0], -1)  # (E, F)

    # 每组的 max（数值稳定），scatter_reduce amax 需 torch >= 1.12
    max_per_node = torch.full((graph.num_nodes, logits_flat.shape[1]),
                              float('-inf'), device=logits.device, dtype=logits.dtype)
    max_per_node.scatter_reduce_(0, tgt.unsqueeze(1).expand_as(logits_flat),
                                 logits_flat, reduce='amax', include_self=False)

    logits_stable = logits_flat - max_per_node[tgt]
    exp = torch.exp(logits_stable)

    sum_per_node = torch.zeros((graph.num_nodes, logits_flat.shape[1]),
                               device=logits.device, dtype=logits.dtype)
    sum_per_node.scatter_add_(0, tgt.unsqueeze(1).expand_as(exp), exp)

    out = exp / sum_per_node[tgt]
    return out.reshape(shape)


class AvgPooling(nn.Module):
    """图级平均池化（dgl.nn.AvgPooling 的简化等价实现）。"""

    def forward(self, graph, feat, *args, **kwargs):
        return feat.mean(dim=0)


class MaxPooling(nn.Module):
    """图级最大池化（dgl.nn.MaxPooling 的简化等价实现）。"""

    def forward(self, graph, feat, *args, **kwargs):
        return feat.max(dim=0)[0]