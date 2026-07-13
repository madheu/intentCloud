"""常识加载器单元测试。

覆盖场景：
  a. 加载后 cloud._shell 中有新节点
  b. 加载后 cloud._edges 中有新边
  c. 带 embedding 的节点 llm_embedding 不为 None
  d. 不提供 model 时，无 embedding 字段的节点 llm_embedding 为 None
  e. 边的 ref_weight = weight
  f. 加载的节点是可演化的（is_node_mutable 返回 True）
  g. 返回值 = (节点数, 边数)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from core.intent_cloud import IntentCloud, CloudEdge
from core.models import ImmutableKernel, Identity, Constraint


# ── Fixtures ──────────────────────────────────────────────────────────────────

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def test_data_path() -> str:
    return str(FIXTURE_DIR / "common_sense_test.json")


@pytest.fixture
def cloud() -> IntentCloud:
    """标准 IntentCloud（无额外节点）。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[Constraint(id="c1", text="safe", level="absolute")],
    )
    return IntentCloud(kernel=kernel)


# ── 场景 a：加载后 cloud._shell 中有新节点 ────────────────────────────────────

def test_load_creates_nodes(cloud, test_data_path):
    """加载后 cloud._shell 中有新节点。"""
    initial_shell_size = len(cloud._shell)
    cloud.load_common_sense(test_data_path)

    assert len(cloud._shell) == initial_shell_size + 3
    assert "time_past" in cloud._shell
    assert "time_future" in cloud._shell
    assert "space_here" in cloud._shell

    # 验证节点字段
    node = cloud._shell["time_past"]
    assert node.text == "过去"
    assert node.trust == 0.6


# ── 场景 b：加载后 cloud._edges 中有新边 ───────────────────────────────────────

def test_load_creates_edges(cloud, test_data_path):
    """加载后 cloud._edges 中有新边。"""
    initial_edge_count = len(cloud._edges)
    cloud.load_common_sense(test_data_path)

    assert len(cloud._edges) == initial_edge_count + 2
    assert ("time_past", "time_future") in cloud._edges
    assert ("time_past", "space_here") in cloud._edges

    edge = cloud._edges[("time_past", "time_future")]
    assert edge.edge_type == "contrasts"
    assert edge.weight == 0.2


# ── 场景 c：带 embedding 的节点 llm_embedding 不为 None ────────────────────────

def test_load_with_embeddings(cloud, test_data_path):
    """带 embedding 的节点 llm_embedding 不为 None。"""
    cloud.load_common_sense(test_data_path)

    # time_past 有 embedding
    node = cloud._shell["time_past"]
    assert node.llm_embedding is not None
    assert isinstance(node.llm_embedding, torch.Tensor)
    assert torch.allclose(
        node.llm_embedding, torch.tensor([0.1, -0.2, 0.3]), atol=1e-6
    )


# ── 场景 d：不提供 model 时无 embedding 的节点 llm_embedding 为 None ──────────

def test_load_without_model(cloud, test_data_path):
    """不提供 model 时，无 embedding 字段的节点 llm_embedding 为 None。"""
    cloud.load_common_sense(test_data_path)

    # time_future 没有 embedding 字段，也没有 model → None
    node = cloud._shell["time_future"]
    assert node.llm_embedding is None


# ── 场景 e：边的 ref_weight = weight ──────────────────────────────────────────

def test_load_ref_weight_equals_weight(cloud, test_data_path):
    """边的 ref_weight = weight。"""
    cloud.load_common_sense(test_data_path)

    for (src, tgt), edge in cloud._edges.items():
        # 只检查新加载的边（非锚点边）
        if not cloud.anchor_system.is_edge_mutable(src, tgt):
            continue
        assert edge.ref_weight == edge.weight, (
            f"Edge {src}->{tgt}: ref_weight={edge.ref_weight} != weight={edge.weight}"
        )


# ── 场景 f：加载的节点是可演化的 ──────────────────────────────────────────────

def test_load_nodes_are_mutable(cloud, test_data_path):
    """加载的节点是可演化的（is_node_mutable 返回 True）。"""
    cloud.load_common_sense(test_data_path)

    for node_id in ["time_past", "time_future", "space_here"]:
        assert cloud.anchor_system.is_node_mutable(node_id), (
            f"Node {node_id} should be mutable"
        )


# ── 场景 g：返回值 = (节点数, 边数) ───────────────────────────────────────────

def test_load_returns_counts(cloud, test_data_path):
    """返回值 = (节点数, 边数)。"""
    n_nodes, n_edges = cloud.load_common_sense(test_data_path)

    assert n_nodes == 3
    assert n_edges == 2


# ── 边界：文件不存在 ──────────────────────────────────────────────────────────

def test_load_file_not_found(cloud):
    """不存在的文件抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        cloud.load_common_sense("nonexistent_file.json")


# ── 边界：空节点/边列表 ───────────────────────────────────────────────────────

def test_load_empty_json(cloud, tmp_path):
    """空节点和边列表不报错。"""
    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps({"version": "1.0", "nodes": [], "edges": []}))

    n_nodes, n_edges = cloud.load_common_sense(str(empty_path))
    assert n_nodes == 0
    assert n_edges == 0


# ── 便捷方法：cloud.load_common_sense() 等同于直接调用 ─────────────────────────

def test_convenience_method_equivalent(cloud, test_data_path):
    """cloud.load_common_sense() 与 load_common_sense(cloud, ...) 行为一致。"""
    from core.common_sense_loader import load_common_sense

    cloud1 = IntentCloud(
        kernel=ImmutableKernel(
            identity=Identity(name="TestBot", role="test", immutable=True),
            safety_constraints=[Constraint(id="c1", text="safe", level="absolute")],
        )
    )
    cloud2 = IntentCloud(
        kernel=ImmutableKernel(
            identity=Identity(name="TestBot", role="test", immutable=True),
            safety_constraints=[Constraint(id="c1", text="safe", level="absolute")],
        )
    )

    n1, e1 = cloud1.load_common_sense(test_data_path)
    n2, e2 = load_common_sense(cloud2, test_data_path)

    assert n1 == n2
    assert e1 == e2
    assert len(cloud1._shell) == len(cloud2._shell)
    assert len(cloud1._edges) == len(cloud2._edges)