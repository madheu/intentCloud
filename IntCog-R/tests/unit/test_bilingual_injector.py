"""双语者方案单元测试。

覆盖：
  a. BilingualInjector.inject() — 正确形状和排序
  b. inject() — 空节点返回 None
  c. inject() — 低于阈值被过滤
  d. inject() — 无 llm_embedding 被跳过
  e. 激活值加权排序验证
  f. generate_with_injection() — 无虚拟 token 时正常生成（降级）
  g. generate_with_injection() — 有虚拟 token 时正常生成
  h. get_node_embedding() — 正常获取 embedding
  i. get_node_embedding() — 空文本报错
  j. get_node_embedding() — 越界 layer_idx 报错
  k. IntentNode llm_embedding 字段存在且默认 None
  l. 锚点节点也可以有 embedding
"""

from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock, patch

from core.models import IntentNode, ImmutableKernel, Identity, Constraint
from core.anchor_system import AnchorSystem, AnchorNodeConfig
from core.intent_cloud import IntentCloud


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cloud_with_nodes() -> IntentCloud:
    """创建一个带节点的 IntentCloud。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[Constraint(id="c1", text="safe", level="absolute")],
    )
    cloud = IntentCloud(kernel=kernel)

    # 添加外壳节点
    cloud._shell["sea"] = IntentNode(
        id="sea", text="大海", trust=0.8, strength=1.0,
        llm_embedding=torch.tensor([1.0, 0.0, 0.0]),
    )
    cloud._shell["sky"] = IntentNode(
        id="sky", text="星空", trust=0.7, strength=0.9,
        llm_embedding=torch.tensor([0.0, 1.0, 0.0]),
    )
    cloud._shell["poem"] = IntentNode(
        id="poem", text="诗歌", trust=0.9, strength=1.0,
        llm_embedding=torch.tensor([0.0, 0.0, 1.0]),
    )
    # 无 embedding 的节点
    cloud._shell["no_emb"] = IntentNode(
        id="no_emb", text="无embedding", trust=0.5,
    )

    return cloud


def _make_mock_model():
    """工厂函数，创建模拟 HuggingFace 模型。"""
    model = MagicMock()
    # 模拟 get_input_embeddings
    embed_layer = MagicMock()
    embed_layer.return_value = torch.randn(1, 5, 3)
    model.get_input_embeddings.return_value = embed_layer
    # 模拟 generate
    model.generate.return_value = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    # 模拟 parameters
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.return_value = iter([param])
    return model


def _make_mock_tokenizer():
    """工厂函数，创建模拟 tokenizer。"""
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    tok.decode.return_value = "生成的诗歌文本"
    tok.eos_token_id = 0
    return tok


# ── 场景 a：inject() 正确形状和排序 ────────────────────────────────────────────

def test_inject_correct_shape(cloud_with_nodes):
    """3 个激活节点 → inject 返回 [3, 3] 形状。"""
    from core.bilingual_injector import BilingualInjector

    injector = BilingualInjector(_make_mock_model(), _make_mock_tokenizer())
    activations = {"sea": 0.5, "sky": 0.8, "poem": 0.3}

    result = injector.inject(activations, cloud_with_nodes)

    assert result is not None
    assert result.shape == (3, 3)  # 3 个节点 × 3 维 embedding


def test_inject_sorted_by_activation(cloud_with_nodes):
    """激活值高的排前面。"""
    from core.bilingual_injector import BilingualInjector

    injector = BilingualInjector(_make_mock_model(), _make_mock_tokenizer())
    # sky=0.8 should be first, sea=0.5 second, poem=0.3 third
    activations = {"sea": 0.5, "sky": 0.8, "poem": 0.3}

    result = injector.inject(activations, cloud_with_nodes)

    assert result is not None
    # sky 的 embedding = [0, 1, 0] × 0.8 = [0, 0.8, 0]，应该排第一
    assert torch.allclose(result[0], torch.tensor([0.0, 0.8, 0.0]), atol=1e-6)
    # sea 的 embedding = [1, 0, 0] × 0.5 = [0.5, 0, 0]，排第二
    assert torch.allclose(result[1], torch.tensor([0.5, 0.0, 0.0]), atol=1e-6)
    # poem 的 embedding = [0, 0, 1] × 0.3 = [0, 0, 0.3]，排第三
    assert torch.allclose(result[2], torch.tensor([0.0, 0.0, 0.3]), atol=1e-6)


# ── 场景 b：空节点返回 None ────────────────────────────────────────────────────

def test_inject_no_activations(cloud_with_nodes):
    """空激活字典返回 None。"""
    from core.bilingual_injector import BilingualInjector

    injector = BilingualInjector(_make_mock_model(), _make_mock_tokenizer())
    result = injector.inject({}, cloud_with_nodes)
    assert result is None


def test_inject_nonexistent_nodes(cloud_with_nodes):
    """激活字典中不存在于 cloud 的节点被跳过 → 返回 None。"""
    from core.bilingual_injector import BilingualInjector

    injector = BilingualInjector(_make_mock_model(), _make_mock_tokenizer())
    result = injector.inject({"ghost": 0.9}, cloud_with_nodes)
    assert result is None


# ── 场景 c：低于阈值被过滤 ─────────────────────────────────────────────────────

def test_inject_below_threshold_filtered(cloud_with_nodes):
    """激活值低于阈值的节点被过滤。"""
    from core.bilingual_injector import BilingualInjector

    # 阈值设为 0.5
    injector = BilingualInjector(_make_mock_model(), _make_mock_tokenizer(), activation_threshold=0.5)
    # sea=0.5 刚好等于阈值（应包含），poem=0.3 低于阈值（应过滤）
    activations = {"sea": 0.5, "poem": 0.3}

    result = injector.inject(activations, cloud_with_nodes)

    assert result is not None
    assert result.shape[0] == 1  # 只有 sea


def test_inject_threshold_zero(cloud_with_nodes):
    """阈值为 0 时所有非零激活节点都参与。"""
    from core.bilingual_injector import BilingualInjector

    injector = BilingualInjector(_make_mock_model(), _make_mock_tokenizer(), activation_threshold=0.0)
    activations = {"sea": 0.01, "sky": 0.001, "poem": 0.0001}

    result = injector.inject(activations, cloud_with_nodes)

    assert result is not None
    assert result.shape[0] == 3  # 全部参与


# ── 场景 d：无 llm_embedding 被跳过 ────────────────────────────────────────────

def test_inject_skip_no_embedding(cloud_with_nodes):
    """无 llm_embedding 的节点被跳过。"""
    from core.bilingual_injector import BilingualInjector

    injector = BilingualInjector(_make_mock_model(), _make_mock_tokenizer())
    # no_emb 没有 llm_embedding
    activations = {"sea": 0.9, "no_emb": 0.9}

    result = injector.inject(activations, cloud_with_nodes)

    assert result is not None
    assert result.shape[0] == 1  # 只有 sea


# ── 场景 e：激活值加权 ─────────────────────────────────────────────────────────

def test_inject_activation_weighting(cloud_with_nodes):
    """embedding × 激活值正确加权。"""
    from core.bilingual_injector import BilingualInjector

    injector = BilingualInjector(_make_mock_model(), _make_mock_tokenizer())
    # sea: [1, 0, 0] × 0.5 = [0.5, 0, 0]
    activations = {"sea": 0.5}

    result = injector.inject(activations, cloud_with_nodes)

    assert result is not None
    assert torch.allclose(result[0], torch.tensor([0.5, 0.0, 0.0]), atol=1e-6)


# ── 场景 f：generate_with_injection 无虚拟 token 降级 ──────────────────────────

def test_generate_no_virtual_tokens(cloud_with_nodes):
    """无激活节点时退化为普通生成。"""
    from core.bilingual_injector import BilingualInjector

    model = _make_mock_model()
    injector = BilingualInjector(model, _make_mock_tokenizer())

    result = injector.generate_with_injection(
        "写一首诗", {}, cloud_with_nodes, max_new_tokens=50
    )

    assert isinstance(result, str)
    assert len(result) > 0
    # 应调用 model.generate 且使用 input_ids
    model.generate.assert_called_once()


# ── 场景 g：generate_with_injection 有虚拟 token ───────────────────────────────

def test_generate_with_virtual_tokens(cloud_with_nodes):
    """有激活节点时 prepend 虚拟 token 后生成。"""
    from core.bilingual_injector import BilingualInjector

    model = _make_mock_model()
    injector = BilingualInjector(model, _make_mock_tokenizer())
    activations = {"sea": 0.5, "sky": 0.8}

    result = injector.generate_with_injection(
        "写一首诗", activations, cloud_with_nodes, max_new_tokens=50
    )

    assert isinstance(result, str)
    assert len(result) > 0
    # 应使用 inputs_embeds=
    call_kwargs = model.generate.call_args[1]
    assert "inputs_embeds" in call_kwargs


# ── 场景 h：get_node_embedding ─────────────────────────────────────────────────

def test_get_node_embedding_basic():
    """正常获取概念节点的 embedding。"""
    from core.node_embedding import get_node_embedding

    # 用 TinyTransformer 作为测试模型
    from mvp_steering_demo import TinyTransformer

    model = TinyTransformer(d_model=64, n_layers=4, vocab_size=128)
    # 模拟 tokenizer
    class FakeTokenizer:
        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([[min(ord(c) % 128, 127) for c in text]])
            return {"input_ids": ids}

    tok = FakeTokenizer()

    emb = get_node_embedding("大海", model, tok, layer_idx=2)
    assert emb.shape == (64,)
    assert emb.dtype == torch.float32


def test_get_node_embedding_empty_text():
    """空文本抛出 ValueError。"""
    from core.node_embedding import get_node_embedding, _get_hidden_dim, _get_num_layers

    with pytest.raises(ValueError, match="must not be empty"):
        get_node_embedding("", MagicMock(), MagicMock(), 0)


def test_get_node_embedding_layer_out_of_range():
    """层索引越界抛出 ValueError。"""
    from core.node_embedding import get_node_embedding
    from mvp_steering_demo import TinyTransformer

    model = TinyTransformer(d_model=64, n_layers=4, vocab_size=128)

    with pytest.raises(ValueError, match="out of range"):
        get_node_embedding("test", model, MagicMock(), layer_idx=999)


# ── 场景 k：IntentNode 默认 llm_embedding 为 None ──────────────────────────────

def test_intent_node_llm_embedding_default():
    """新建 IntentNode 时 llm_embedding 默认为 None。"""
    node = IntentNode(id="test", text="测试")
    assert node.llm_embedding is None


def test_intent_node_llm_embedding_set():
    """可以设置 llm_embedding 为 torch.Tensor。"""
    emb = torch.tensor([1.0, 2.0, 3.0])
    node = IntentNode(id="test", text="测试", llm_embedding=emb)
    assert torch.equal(node.llm_embedding, emb)


# ── 场景 l：锚点节点也可以有 embedding ─────────────────────────────────────────

def test_anchor_node_can_have_embedding():
    """锚点节点配置支持 llm_embedding。"""
    from core.models import IntentNode

    # 锚点节点可以是 IntentNode 的子类或用其字段
    node = IntentNode(
        id="self", text="系统自身", trust=1.0, strength=1.0,
        llm_embedding=torch.tensor([0.5, 0.5, 0.5]),
    )
    assert node.llm_embedding is not None
    assert torch.equal(node.llm_embedding, torch.tensor([0.5, 0.5, 0.5]))


# ── 辅助函数测试 ──────────────────────────────────────────────────────────────

def test_get_hidden_dim():
    """正确获取 hidden_dim。"""
    from core.node_embedding import _get_hidden_dim

    model = MagicMock()
    model.config.hidden_size = 768
    assert _get_hidden_dim(model) == 768

    model2 = MagicMock()
    del model2.config.hidden_size
    model2.config.d_model = 512
    assert _get_hidden_dim(model2) == 512


def test_get_num_layers():
    """正确获取层数。"""
    from core.node_embedding import _get_num_layers

    model = MagicMock()
    model.config.num_hidden_layers = 24
    assert _get_num_layers(model) == 24


def test_get_layer_llama_style():
    """LLaMA 风格模型结构。"""
    from core.node_embedding import _get_layer

    model = MagicMock()
    model.model.layers = [MagicMock(), MagicMock(), MagicMock()]
    assert _get_layer(model, 1) is model.model.layers[1]


def test_get_layer_gpt2_style():
    """GPT-2 风格模型结构。"""
    from core.node_embedding import _get_layer

    model = MagicMock()
    del model.model  # 确保没有 LLaMA 属性
    model.transformer.h = [MagicMock(), MagicMock()]
    assert _get_layer(model, 0) is model.transformer.h[0]


def test_get_status():
    """get_status 返回正确状态。"""
    from core.bilingual_injector import BilingualInjector

    model = _make_mock_model()
    injector = BilingualInjector(model, _make_mock_tokenizer(), activation_threshold=0.3)
    status = injector.get_status()
    assert status["activation_threshold"] == 0.3
    assert "model_type" in status