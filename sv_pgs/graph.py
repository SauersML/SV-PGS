from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs.config import ModelConfig
from sv_pgs.data import GraphEdges, TieMap, VariantRecord


@dataclass(slots=True)
class CorrelationBlock:
    indices: np.ndarray


def build_correlation_graph(
    genotypes: np.ndarray,
    records: Sequence[VariantRecord],
    tie_map: TieMap,
    config: ModelConfig,
) -> GraphEdges:
    reduced_x = np.asarray(genotypes[:, tie_map.kept_indices], dtype=np.float32)
    reduced_records = [records[index] for index in tie_map.kept_indices]
    edge_map: dict[tuple[int, int], tuple[float, float]] = {}

    for left in range(len(reduced_records)):
        left_record = reduced_records[left]
        for right in range(left + 1, len(reduced_records)):
            right_record = reduced_records[right]
            if not _eligible_pair(left_record, right_record, config):
                continue
            corr = _safe_corr(reduced_x[:, left], reduced_x[:, right])
            abs_corr = abs(corr)
            if abs_corr < config.correlation_threshold:
                continue
            sign = 1.0 if corr >= 0.0 else -1.0
            strength = config.same_class_edge_strength if left_record.variant_class == right_record.variant_class else config.cross_class_edge_strength
            weight = strength * abs_corr
            edge_map[(left, right)] = (sign, weight)

    src: list[int] = []
    dst: list[int] = []
    sign: list[float] = []
    weight: list[float] = []
    adjacency: dict[int, list[int]] = defaultdict(list)
    for (left, right), (edge_sign, edge_weight) in edge_map.items():
        src.append(left)
        dst.append(right)
        sign.append(edge_sign)
        weight.append(edge_weight)
        adjacency[left].append(right)
        adjacency[right].append(left)

    block_ids = np.full(reduced_x.shape[1], -1, dtype=np.int32)
    block_index = 0
    for node in range(reduced_x.shape[1]):
        if block_ids[node] != -1:
            continue
        stack = [node]
        block_ids[node] = block_index
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if block_ids[neighbor] != -1:
                    continue
                block_ids[neighbor] = block_index
                stack.append(neighbor)
        block_index += 1

    return GraphEdges(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        sign=np.asarray(sign, dtype=np.float32),
        weight=np.asarray(weight, dtype=np.float32),
        block_ids=block_ids,
    )


def correlation_blocks(edges: GraphEdges) -> list[CorrelationBlock]:
    blocks: list[CorrelationBlock] = []
    for block_id in np.unique(edges.block_ids):
        members = np.where(edges.block_ids == block_id)[0]
        blocks.append(CorrelationBlock(indices=members.astype(np.int32)))
    return blocks


def _eligible_pair(left: VariantRecord, right: VariantRecord, config: ModelConfig) -> bool:
    if left.cluster_id and left.cluster_id == right.cluster_id:
        return True
    if left.chromosome != right.chromosome:
        return False
    return abs(left.position - right.position) <= config.graph_window_bp


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
