import torch
from typing import Tuple
from torch_scatter import scatter
from typing import List, Optional, Tuple, Union
from torch import Tensor
from copy import copy
from torch_geometric.typing import SparseTensor  # noqa


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def maybe_num_nodes_dict(edge_index_dict, num_nodes_dict=None):
    num_nodes_dict = {} if num_nodes_dict is None else copy(num_nodes_dict)

    found_types = list(num_nodes_dict.keys())

    for keys, edge_index in edge_index_dict.items():
        key = keys[0]
        if key not in found_types:
            N = int(edge_index[0].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        key = keys[-1]
        if key not in found_types:
            N = int(edge_index[1].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

    return num_nodes_dict


@torch.jit._overload
def coalesce(
    edge_index,
    edge_attr=None,
    num_nodes=None,
    reduce="add",
    is_sorted=False,
    sort_by_row=True,
):
    # type: (Tensor, Optional[bool], Optional[int], str, bool, bool) -> Tensor  # noqa
    pass


@torch.jit._overload
def coalesce(
    edge_index,
    edge_attr=None,
    num_nodes=None,
    reduce="add",
    is_sorted=False,
    sort_by_row=True,
):
    # type: (Tensor, Tensor, Optional[int], str, bool, bool) -> Tuple[Tensor, Tensor]  # noqa
    pass


@torch.jit._overload
def coalesce(
    edge_index,
    edge_attr=None,
    num_nodes=None,
    reduce="add",
    is_sorted=False,
    sort_by_row=True,
):
    # type: (Tensor, List[Tensor], Optional[int], str, bool, bool) -> Tuple[Tensor, List[Tensor]]  # noqa
    pass


def coalesce(
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor]] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    nnz = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        if isinstance(edge_attr, (Tensor, list, tuple)):
            return edge_index, edge_attr
        return edge_index

    edge_index = edge_index[:, mask]

    if edge_attr is None:
        return edge_index

    dim_size = edge_index.size(1)
    idx = torch.arange(0, nnz, device=edge_index.device)
    idx.sub_(mask.logical_not_().cumsum(dim=0))

    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, None, dim_size, reduce)
        return edge_index, edge_attr
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [scatter(e, idx, 0, None, dim_size, reduce) for e in edge_attr]
        return edge_index, edge_attr

    return edge_index


@torch.jit._overload
def to_undirected(edge_index, edge_attr=None, num_nodes=None, reduce="add"):
    # type: (Tensor, Optional[bool], Optional[int], str) -> Tensor  # noqa
    pass


@torch.jit._overload
def to_undirected(edge_index, edge_attr=None, num_nodes=None, reduce="add"):
    # type: (Tensor, Tensor, Optional[int], str) -> Tuple[Tensor, Tensor]  # noqa
    pass


@torch.jit._overload
def to_undirected(edge_index, edge_attr=None, num_nodes=None, reduce="add"):
    # type: (Tensor, List[Tensor], Optional[int], str) -> Tuple[Tensor, List[Tensor]]  # noqa
    pass


def to_undirected(
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor]] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    row, col = edge_index[0], edge_index[1]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    if isinstance(edge_attr, Tensor):
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attr = [torch.cat([e, e], dim=0) for e in edge_attr]

    return coalesce(edge_index, edge_attr, num_nodes, reduce)
