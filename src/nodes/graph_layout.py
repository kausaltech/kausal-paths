from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from statistics import fmean
from typing import TYPE_CHECKING

import strawberry as sb

import networkx as nx

if TYPE_CHECKING:
    from nodes.context import Context
    from nodes.node import Node


@sb.enum
class PrimaryLayoutClass(Enum):
    ACTION = 'action'
    OUTCOME = 'outcome'
    CORE = 'core'
    CONTEXT_SOURCE = 'context_source'
    GHOSTABLE_CONTEXT_SOURCE = 'ghostable_context_source'


@sb.type
class GraphLayoutThresholds:
    hub_degree: int = 7
    ghostable_out_degree: int = 2
    ghostable_total_degree: int = 5
    ghostable_avg_outgoing_span: float = 3.0


@sb.type
class GraphLayout:
    thresholds: GraphLayoutThresholds
    core_node_ids: list[sb.ID]
    ghostable_context_source_ids: list[sb.ID]
    hub_ids: list[sb.ID]
    action_ids: list[sb.ID]
    outcome_ids: list[sb.ID]
    main_graph_node_ids: list[sb.ID]


@sb.type
class NodeGraphLayoutMeta:
    node_id: sb.ID
    primary_class: PrimaryLayoutClass
    is_hub: bool
    has_action_ancestor: bool
    topological_layer: int
    in_degree: int
    out_degree: int
    total_degree: int
    avg_outgoing_span: float
    max_outgoing_span: int
    ghostable: bool
    ghost_targets: list[sb.ID]
    canonical_rail: str | None


class NodeGraphClassifier:
    def __init__(self, context: Context, thresholds: GraphLayoutThresholds | None = None):
        self.context = context
        self.thresholds = thresholds or GraphLayoutThresholds()
        self.graph = context.node_graph
        self.nodes = context.nodes

        self._generations = list(nx.topological_generations(self.graph))
        self._topological_layer = {node_id: idx for idx, generation in enumerate(self._generations) for node_id in generation}
        self._has_action_ancestor = self._compute_action_ancestry()
        self._metadata = self._build_metadata()

        self.actions = self._ids_for_class(PrimaryLayoutClass.ACTION)
        self.outcomes = self._ids_for_class(PrimaryLayoutClass.OUTCOME)
        self.core_nodes = self._ids_for_class(PrimaryLayoutClass.CORE)
        self.context_sources = self._ids_for_class(PrimaryLayoutClass.CONTEXT_SOURCE)
        self.ghostable_context_sources = self._ids_for_class(PrimaryLayoutClass.GHOSTABLE_CONTEXT_SOURCE)
        self.hubs = self._sorted_ids(node_id for node_id, meta in self._metadata.items() if meta.is_hub)
        self.main_graph_node_ids = self._sorted_ids(
            node_id
            for node_id, meta in self._metadata.items()
            if meta.primary_class is not PrimaryLayoutClass.GHOSTABLE_CONTEXT_SOURCE
        )

    @property
    def metadata(self) -> dict[str, NodeGraphLayoutMeta]:
        return self._metadata

    def for_node(self, node_id: str) -> NodeGraphLayoutMeta:
        return self._metadata[node_id]

    def _compute_action_ancestry(self) -> dict[str, bool]:
        from nodes.actions.action import ActionNode

        ancestry: dict[str, bool] = {}
        for generation in self._generations:
            for node_id in generation:
                preds = self.graph.predecessors(node_id)
                ancestry[node_id] = any(isinstance(self.nodes[pred_id], ActionNode) or ancestry[pred_id] for pred_id in preds)
        return ancestry

    def _build_metadata(self) -> dict[str, NodeGraphLayoutMeta]:
        metadata: dict[str, NodeGraphLayoutMeta] = {}
        for node_id in self.graph.nodes:
            node = self.nodes[node_id]
            in_degree = self.graph.in_degree(node_id)
            out_degree = self.graph.out_degree(node_id)
            total_degree = self.graph.degree(node_id)
            outgoing_spans = [
                self._topological_layer[target_id] - self._topological_layer[node_id]
                for target_id in self.graph.successors(node_id)
            ]
            avg_outgoing_span = fmean(outgoing_spans) if outgoing_spans else 0.0
            max_outgoing_span = max(outgoing_spans, default=0)
            has_action_ancestor = self._has_action_ancestor[node_id]
            primary_class = self._classify_primary(
                node=node,
                has_action_ancestor=has_action_ancestor,
                in_degree=in_degree,
                out_degree=out_degree,
                total_degree=total_degree,
                avg_outgoing_span=avg_outgoing_span,
            )
            ghostable = primary_class is PrimaryLayoutClass.GHOSTABLE_CONTEXT_SOURCE
            metadata[node_id] = NodeGraphLayoutMeta(
                node_id=sb.ID(node_id),
                primary_class=primary_class,
                is_hub=total_degree >= self.thresholds.hub_degree,
                has_action_ancestor=has_action_ancestor,
                topological_layer=self._topological_layer[node_id],
                in_degree=in_degree,
                out_degree=out_degree,
                total_degree=total_degree,
                avg_outgoing_span=avg_outgoing_span,
                max_outgoing_span=max_outgoing_span,
                ghostable=ghostable,
                ghost_targets=(
                    [sb.ID(node_id) for node_id in self._sorted_ids(self.graph.successors(node_id))] if ghostable else []
                ),
                canonical_rail='context' if ghostable else None,
            )
        return metadata

    def _classify_primary(
        self,
        *,
        node: Node,
        has_action_ancestor: bool,
        in_degree: int,
        out_degree: int,
        total_degree: int,
        avg_outgoing_span: float,
    ) -> PrimaryLayoutClass:
        from nodes.actions.action import ActionNode

        if isinstance(node, ActionNode):
            return PrimaryLayoutClass.ACTION
        if node.is_outcome:
            return PrimaryLayoutClass.OUTCOME
        if not has_action_ancestor:
            if (
                out_degree >= self.thresholds.ghostable_out_degree
                or total_degree >= self.thresholds.ghostable_total_degree
                or avg_outgoing_span >= self.thresholds.ghostable_avg_outgoing_span
                or (in_degree == 0 and out_degree >= 1)
            ):
                return PrimaryLayoutClass.GHOSTABLE_CONTEXT_SOURCE
            return PrimaryLayoutClass.CONTEXT_SOURCE
        return PrimaryLayoutClass.CORE

    def _ids_for_class(self, primary_class: PrimaryLayoutClass) -> list[str]:
        return self._sorted_ids(node_id for node_id, meta in self._metadata.items() if meta.primary_class == primary_class)

    def _sorted_ids(self, node_ids) -> list[str]:
        return sorted(node_ids, key=lambda node_id: (self._topological_layer.get(node_id, 0), node_id))


@dataclass(frozen=True)
class GraphClusterThresholds:
    action_signature_horizon: int = 4
    action_overlap_min_shared_descendants: int = 2
    action_overlap_min_overlap: float = 0.2
    successor_influence_weight: float = 0.35
    node_group_bonus: float = 0.75
    bridge_score_threshold: float = 0.3


@dataclass(frozen=True)
class NodeGraphClusterMeta:
    node_id: str
    cluster_id: str | None
    cluster_label: str | None
    cluster_confidence: float
    seed_action_ids: tuple[str, ...]
    neighboring_clusters: tuple[str, ...]
    bridge_score: float
    is_bridge: bool


@dataclass(frozen=True)
class NodeGraphCluster:
    id: str
    label: str
    action_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    bridge_node_ids: tuple[str, ...]
    incoming_cluster_ids: tuple[str, ...]
    outgoing_cluster_ids: tuple[str, ...]
    node_group_hints: tuple[str, ...]


@dataclass(frozen=True)
class _ActionClusterSeed:
    id: str
    label: str
    action_ids: tuple[str, ...]
    normalized_tokens: tuple[str, ...]
    node_group_hints: tuple[str, ...]


class NodeGraphClusterer:
    def __init__(self, context: Context, thresholds: GraphClusterThresholds | None = None):
        self.context = context
        self.thresholds = thresholds or GraphClusterThresholds()
        self.graph = context.node_graph
        self.nodes = context.nodes
        self.classifier = context.node_graph_classifier

        self._generations = list(nx.topological_generations(self.graph))
        self._topological_layer = {node_id: idx for idx, generation in enumerate(self._generations) for node_id in generation}
        self._action_ids = tuple(self.classifier.actions)
        self._descendant_sets = self._compute_action_descendant_sets()
        self._seeds = self._build_action_cluster_seeds()
        self.cluster_ids = [seed.id for seed in self._seeds]
        self._seed_by_id = {seed.id: seed for seed in self._seeds}
        self.action_cluster_ids = {action_id: seed.id for seed in self._seeds for action_id in seed.action_ids}
        self._cluster_tokens = {seed.id: set(seed.normalized_tokens) for seed in self._seeds}
        self._upstream_action_sets = self._compute_upstream_action_sets()
        self._score_vectors = self._build_score_vectors()
        self._metadata = self._build_metadata()
        self.inter_cluster_edges = self._build_inter_cluster_edges()
        self.clusters = self._build_clusters()

    @property
    def metadata(self) -> dict[str, NodeGraphClusterMeta]:
        return self._metadata

    def for_node(self, node_id: str) -> NodeGraphClusterMeta:
        return self._metadata[node_id]

    def summarize(self) -> list[dict[str, object]]:
        return [
            {
                'id': cluster.id,
                'label': cluster.label,
                'actions': list(cluster.action_ids),
                'nodes': len(cluster.node_ids),
                'bridges': list(cluster.bridge_node_ids),
                'incoming': list(cluster.incoming_cluster_ids),
                'outgoing': list(cluster.outgoing_cluster_ids),
                'node_group_hints': list(cluster.node_group_hints),
            }
            for cluster in self.clusters.values()
        ]

    def _compute_action_descendant_sets(self) -> dict[str, set[str]]:
        descendant_sets: dict[str, set[str]] = {}
        for action_id in self._action_ids:
            lengths = nx.single_source_shortest_path_length(
                self.graph,
                action_id,
                cutoff=self.thresholds.action_signature_horizon,
            )
            signature: set[str] = set()
            for node_id, distance in lengths.items():
                if distance == 0:
                    continue
                layout_meta = self.classifier.for_node(node_id)
                if layout_meta.primary_class is PrimaryLayoutClass.OUTCOME or layout_meta.is_hub:
                    continue
                signature.add(node_id)
            descendant_sets[action_id] = signature
        return descendant_sets

    def _build_action_cluster_seeds(self) -> list[_ActionClusterSeed]:
        if not self._action_ids:
            return []

        action_graph: nx.Graph[str] = nx.Graph()
        action_graph.add_nodes_from(self._action_ids)

        for idx, action_id in enumerate(self._action_ids):
            a_desc = self._descendant_sets[action_id]
            for other_id in self._action_ids[idx + 1 :]:
                b_desc = self._descendant_sets[other_id]
                if not a_desc or not b_desc:
                    continue
                shared = len(a_desc & b_desc)
                overlap = shared / min(len(a_desc), len(b_desc))
                if (
                    shared >= self.thresholds.action_overlap_min_shared_descendants
                    and overlap >= self.thresholds.action_overlap_min_overlap
                ):
                    action_graph.add_edge(action_id, other_id, weight=overlap)

        used_ids: set[str] = set()
        seeds: list[_ActionClusterSeed] = []
        communities: list[set[str]]
        if action_graph.number_of_edges() == 0:
            communities = [{action_id} for action_id in self._action_ids]
        else:
            try:
                communities = [
                    set(component)
                    for component in nx.community.louvain_communities(
                        action_graph,
                        weight='weight',
                        seed=0,
                    )
                ]
            except AttributeError:
                communities = [
                    set(component)
                    for component in nx.community.greedy_modularity_communities(
                        action_graph,
                        weight='weight',
                    )
                ]

        for index, component in enumerate(communities, start=1):
            action_ids = tuple(sorted(component))
            label, normalized_tokens, node_group_hints = self._label_for_actions(action_ids, index=index)
            cluster_id = self._unique_cluster_id(label, index=index, used_ids=used_ids)
            seeds.append(
                _ActionClusterSeed(
                    id=cluster_id,
                    label=label,
                    action_ids=action_ids,
                    normalized_tokens=tuple(sorted(normalized_tokens)),
                    node_group_hints=tuple(sorted(node_group_hints)),
                )
            )
        return seeds

    def _label_for_actions(self, action_ids: tuple[str, ...], *, index: int) -> tuple[str, set[str], set[str]]:
        label_counts: Counter[str] = Counter()
        token_map: dict[str, str] = {}
        node_group_hints: set[str] = set()

        for action_id in action_ids:
            action = self.nodes[action_id]
            group = getattr(action, 'group', None)
            candidates = [
                getattr(group, 'id', None),
                str(getattr(group, 'name', '')).strip() or None,
                action.node_group,
            ]
            if action.node_group:
                node_group_hints.add(action.node_group)
            for candidate in candidates:
                normalized = self._normalize_token(candidate)
                if normalized is None:
                    continue
                label_counts[normalized] += 1
                token_map.setdefault(normalized, candidate or normalized)

        if label_counts:
            normalized_label, _ = label_counts.most_common(1)[0]
            label = token_map[normalized_label]
            normalized_tokens = set(token_map.keys())
            return label, normalized_tokens, node_group_hints

        fallback = action_ids[0] if len(action_ids) == 1 else f'cluster_{index}'
        normalized = self._normalize_token(fallback)
        normalized_tokens = {normalized} if normalized is not None else set()
        return fallback, normalized_tokens, node_group_hints

    def _unique_cluster_id(self, label: str, *, index: int, used_ids: set[str]) -> str:
        base = self._slug(label) or f'cluster_{index}'
        candidate = base
        serial = 2
        while candidate in used_ids:
            candidate = f'{base}_{serial}'
            serial += 1
        used_ids.add(candidate)
        return candidate

    def _compute_upstream_action_sets(self) -> dict[str, set[str]]:
        upstream: dict[str, set[str]] = {}
        action_ids = set(self._action_ids)
        for generation in self._generations:
            for node_id in generation:
                inherited: set[str] = set()
                if node_id in action_ids:
                    inherited.add(node_id)
                for pred_id in self.graph.predecessors(node_id):
                    inherited.update(upstream[pred_id])
                upstream[node_id] = inherited
        return upstream

    def _build_score_vectors(self) -> dict[str, dict[str, float]]:
        raw_scores: dict[str, dict[str, float]] = {}
        for node_id in self.graph.nodes:
            raw_counts = Counter(
                self.action_cluster_ids[action_id]
                for action_id in self._upstream_action_sets[node_id]
                if action_id in self.action_cluster_ids
            )
            raw_scores[node_id] = {cluster_id: float(count) for cluster_id, count in raw_counts.items()}

        score_vectors: dict[str, dict[str, float]] = {}
        for generation in reversed(self._generations):
            for node_id in sorted(generation):
                scores = dict(raw_scores[node_id])
                for succ_id in self.graph.successors(node_id):
                    for cluster_id, value in score_vectors[succ_id].items():
                        scores[cluster_id] = scores.get(cluster_id, 0.0) + (value * self.thresholds.successor_influence_weight)

                normalized_node_group = self._normalize_token(self.nodes[node_id].node_group)
                if normalized_node_group is not None:
                    for cluster_id, tokens in self._cluster_tokens.items():
                        if normalized_node_group in tokens:
                            scores[cluster_id] = scores.get(cluster_id, 0.0) + self.thresholds.node_group_bonus
                score_vectors[node_id] = scores
        return score_vectors

    def _build_metadata(self) -> dict[str, NodeGraphClusterMeta]:
        metadata: dict[str, NodeGraphClusterMeta] = {}
        for node_id in self.graph.nodes:
            cluster_id, confidence = self._pick_cluster(node_id)
            cluster_label = self._seed_by_id[cluster_id].label if cluster_id is not None else None
            neighbor_cluster_ids = {
                meta_cluster_id
                for neighbor_id in (*self.graph.predecessors(node_id), *self.graph.successors(node_id))
                if (meta_cluster_id := self._pick_cluster(neighbor_id)[0]) is not None and meta_cluster_id != cluster_id
            }

            cross_edges = 0
            total_degree = self.graph.degree(node_id)
            if total_degree:
                for pred_id in self.graph.predecessors(node_id):
                    if self._pick_cluster(pred_id)[0] != cluster_id:
                        cross_edges += 1
                for succ_id in self.graph.successors(node_id):
                    if self._pick_cluster(succ_id)[0] != cluster_id:
                        cross_edges += 1

            bridge_score = cross_edges / total_degree if total_degree else 0.0
            seed_action_ids = tuple(
                sorted(
                    action_id
                    for action_id in self._upstream_action_sets[node_id]
                    if cluster_id is not None and self.action_cluster_ids.get(action_id) == cluster_id
                )
            )
            metadata[node_id] = NodeGraphClusterMeta(
                node_id=node_id,
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                cluster_confidence=confidence,
                seed_action_ids=seed_action_ids,
                neighboring_clusters=tuple(sorted(neighbor_cluster_ids)),
                bridge_score=bridge_score,
                is_bridge=bridge_score >= self.thresholds.bridge_score_threshold and bool(neighbor_cluster_ids),
            )
        return metadata

    def _pick_cluster(self, node_id: str) -> tuple[str | None, float]:
        scores = self._score_vectors[node_id]
        if not scores:
            return None, 0.0
        cluster_id, score = max(
            scores.items(),
            key=lambda item: (item[1], -self._topological_layer.get(node_id, 0), item[0]),
        )
        total = sum(scores.values())
        confidence = float(score / total) if total else 0.0
        return cluster_id, confidence

    def _build_inter_cluster_edges(self) -> tuple[tuple[str, str, str, str], ...]:
        inter_edges: list[tuple[str, str, str, str]] = []
        for source_id, target_id in self.graph.edges:
            source_cluster_id = self._metadata[source_id].cluster_id
            target_cluster_id = self._metadata[target_id].cluster_id
            if source_cluster_id is None or target_cluster_id is None or source_cluster_id == target_cluster_id:
                continue
            inter_edges.append((source_id, target_id, source_cluster_id, target_cluster_id))
        return tuple(sorted(inter_edges))

    def _build_clusters(self) -> dict[str, NodeGraphCluster]:
        cluster_nodes: dict[str, list[str]] = {seed.id: [] for seed in self._seeds}
        cluster_bridges: dict[str, list[str]] = {seed.id: [] for seed in self._seeds}
        incoming: dict[str, set[str]] = {seed.id: set() for seed in self._seeds}
        outgoing: dict[str, set[str]] = {seed.id: set() for seed in self._seeds}

        for node_id, meta in self._metadata.items():
            if meta.cluster_id is None:
                continue
            cluster_nodes[meta.cluster_id].append(node_id)
            if meta.is_bridge:
                cluster_bridges[meta.cluster_id].append(node_id)

        for _, _, source_cluster_id, target_cluster_id in self.inter_cluster_edges:
            outgoing[source_cluster_id].add(target_cluster_id)
            incoming[target_cluster_id].add(source_cluster_id)

        clusters: dict[str, NodeGraphCluster] = {}
        for seed in self._seeds:
            clusters[seed.id] = NodeGraphCluster(
                id=seed.id,
                label=seed.label,
                action_ids=seed.action_ids,
                node_ids=tuple(sorted(cluster_nodes[seed.id], key=lambda nid: (self._topological_layer[nid], nid))),
                bridge_node_ids=tuple(sorted(cluster_bridges[seed.id], key=lambda nid: (self._topological_layer[nid], nid))),
                incoming_cluster_ids=tuple(sorted(incoming[seed.id])),
                outgoing_cluster_ids=tuple(sorted(outgoing[seed.id])),
                node_group_hints=seed.node_group_hints,
            )
        return clusters

    @staticmethod
    def _normalize_token(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = re.sub(r'[^a-z0-9]+', '_', value.casefold()).strip('_')
        return normalized or None

    @classmethod
    def _slug(cls, value: str) -> str:
        return cls._normalize_token(value) or ''
