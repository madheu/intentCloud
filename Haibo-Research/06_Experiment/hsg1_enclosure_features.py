"""
H-SG-1 实验：围合特征提取 + 聚类验证
=====================================
假设：同一 Token 在不同上下文中，围合模式可聚类区分义项。

流程：
  输入句子 → 定位目标词 → 提取上下文词 → 查 HowNet 义原
  → 统计义原频次（围合特征向量） → 聚类 → 与金标对比

用法：
  python hsg1_enclosure_features.py --data dev     # 跑 P0-R dev 集
  python hsg1_enclosure_features.py --data llm     # 跑 LLM 语料
  python hsg1_enclosure_features.py --show word    # 查看某词的围合模式
"""
import sys, os, json, re
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# ── 配置 ──
CONTEXT_WINDOW = 3       # 目标词前后各取多少个词
CLUSTER_SEED = 42        # KMeans 随机种子
P0_DATA_DIR = r"E:\intentCloud\Haibo-Research\06_Experiment\data"
LLM_DATA = r"E:\intentCloud\Haibo-Research\06_Experiment\p0_b2_llm_contexts.json"


# ═══════════════════════════════════════════════════════════
# 模块 1：HowNet 接口
# ═══════════════════════════════════════════════════════════

class HowNetInterface:
    """HowNet 义原查询封装。按需加载，只有被调用时才加载模型。"""
    
    _instance = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            print("[HowNet] 加载中...", end=" ", flush=True)
            import OpenHowNet
            cls._instance = cls()
            cls._instance.hownet = OpenHowNet.HowNetDict()
            print("OK")
        return cls._instance
    
    def get_sememes(self, word: str, mfs: bool = True) -> list[str]:
        """获取一个词的义原列表。
        
        Args:
            word: 查询词
            mfs: True=用第一义项，False=返回所有义项的义原
        
        Returns:
            义原名称列表（英文），如 ['Fruit', 'Food', 'Eatable']
        """
        try:
            result = self.hownet.get_sememes_by_word(word, merge=True)
            if not result:
                return []
            # result 是 list[Sememe]，每个 Sememe 有 .zh / .en 属性
            sememes = []
            for s in result:
                zh = getattr(s, 'zh', '') or ''
                en = getattr(s, 'en', '') or ''
                sememes.append(zh or en)
            return sememes
        except Exception as e:
            print(f"  [HowNet WARN] {word}: {e}")
            return []


# ═══════════════════════════════════════════════════════════
# 模块 2：围合特征提取
# ═══════════════════════════════════════════════════════════

class EnclosureFeatureExtractor:
    """从句子中提取目标词的围合特征向量。"""
    
    def __init__(self, window: int = CONTEXT_WINDOW):
        self.window = window
        self.hownet = None  # 延迟加载
        self._vocab: dict[str, int] = {}  # 义原→索引映射
    
    def _ensure_hownet(self):
        if self.hownet is None:
            self.hownet = HowNetInterface.get()
    
    def tokenize(self, sentence: str) -> list[str]:
        """jieba 分词。"""
        import jieba
        return list(jieba.cut(sentence))
    
    def locate_target(self, tokens: list[str], target_word: str, 
                      occurrence: int = 1) -> int | None:
        """定位目标词在 token 列表中的第 occurrence 次出现位置。"""
        count = 0
        for i, tok in enumerate(tokens):
            if target_word in tok or tok == target_word:
                count += 1
                if count == occurrence:
                    return i
        return None
    
    EXCLUDE_WORDS = {"的", "了", "很", "在", "是", "就", "都", "也", "和", "与",
                      "这", "那", "我", "你", "他", "她", "它", "们", "不", "被",
                      "把", "从", "到", "上", "下", "里", "中", "、", "，", "。"}
    
    # 感知词表 — 具有高诊断性的感知词汇（味觉/触觉/听觉/视觉/嗅觉）
    # 这些词的出现强有力地指示了上下文所属的语义域
    PERCEPTUAL_WORDS = {
        # 味觉
        "甜", "酸", "苦", "辣", "咸", "鲜", "涩", "美味", "好吃", "好喝",
        "香甜", "甘甜", "酸甜", "苦涩", "清淡", "浓郁", "入味",
        # 嗅觉
        "香", "臭", "芬芳", "香味", "气味", "味道", "香气",
        # 触觉
        "重", "轻", "软", "硬", "滑", "粗糙", "细腻", "质感", "手感",
        "柔", "韧", "脆", "酥", "弹", "冰凉", "温暖",
        # 视觉
        "红", "绿", "蓝", "黄", "白", "黑", "亮", "暗", "鲜艳",
        "漂亮", "好看", "美观", "精致", "华丽",
        "圆", "方", "大", "小", "长", "短", "厚", "薄",
        # 听觉
        "音质", "声音", "响亮", "清晰", "流畅", "悦耳", "嘈杂", "安静",
        "通话", "音量", "音效",
        # 通用感知
        "感觉", "感知", "体验", "舒服", "舒适", "享受",
    }
    PERCEPTUAL_WEIGHT = 3.0  # 感知词的义原权重倍率
    
    # 感知字符 — 单字感知概念，出现在复合词中也能被检测到
    # 比如"很甜"虽不是完整词，但包含"甜"，仍应视为感知
    PERCEPTUAL_CHARS = set("甜酸苦辣咸鲜涩香臭轻重软硬滑粗糙细腻脆酥弹暖凉"
                           "红绿蓝黄白黑亮暗鲜艳漂亮美观精致华丽"
                           "圆方大小长短厚薄"
                           "响亮清晰流畅悦耳嘈杂安静"
                           "舒服舒适享受")
    
    def extract_context_words(self, tokens: list[str], target_pos: int) -> list[str]:
        """提取目标词周围的上下文词，过滤虚词/功能词。"""
        start = max(0, target_pos - self.window)
        end = min(len(tokens), target_pos + self.window + 1)
        context = []
        for i in range(start, end):
            if i != target_pos:  # 排除目标词本身
                tok = tokens[i]
                if tok not in self.EXCLUDE_WORDS:
                    context.append(tok)
        return context
    
    def extract_features(self, sentence: str, target_word: str,
                         occurrence: int = 1) -> dict:
        """提取一句中目标词的围合特征。
        
        Returns:
            {'word': str, 'sentence': str, 'sememes': Counter, 'vector': list[float]}
        """
        self._ensure_hownet()
        tokens = self.tokenize(sentence)
        pos = self.locate_target(tokens, target_word, occurrence)
        if pos is None:
            return {'word': target_word, 'sentence': sentence, 
                    'sememes': Counter(), 'vector': [], 'status': 'target_not_found'}
        
        context_words = self.extract_context_words(tokens, pos)
        
        # 对每个上下文词查 HowNet → 收集义原
        sememe_counts: Counter = Counter()
        for cw in context_words:
            # 感知检测：整词匹配 OR 字符级匹配
            is_perceptual = (cw in self.PERCEPTUAL_WORDS or 
                            any(c in self.PERCEPTUAL_CHARS for c in cw))
            weight = self.PERCEPTUAL_WEIGHT if is_perceptual else 1.0
            sememes = self.hownet.get_sememes(cw, mfs=True)
            for s in sememes:
                sememe_counts[s] += weight
        
        return {
            'word': target_word,
            'sentence': sentence,
            'context_words': context_words,
            'sememes': dict(sememe_counts.most_common()),
            'n_sememes': len(sememe_counts),
            # 感知特征：上下文中的感知词数量和占比
            'perceptual_count': sum(1 for cw in context_words 
                                    if cw in self.PERCEPTUAL_WORDS 
                                    or any(c in self.PERCEPTUAL_CHARS for c in cw)),
            'perceptual_words': [cw for cw in context_words 
                                if cw in self.PERCEPTUAL_WORDS
                                or any(c in self.PERCEPTUAL_CHARS for c in cw)],
            'n_context': len(context_words),
            'status': 'ok',
        }
    
    def to_vector(self, feature: dict, global_vocab: dict[str, int],
                   mode: str = "hybrid") -> list[float]:
        """将义原计数 + 感知特征映射到向量。
        
        Args:
            mode: 'hybrid' = 语义+感知, 'perceptual_only' = 仅感知, 'sememe_only' = 仅语义
        """
        sememe_counts = feature.get('sememes', {})
        n_vocab = len(global_vocab)
        
        if mode == "perceptual_only":
            # 仅感知特征：感知词绝对数 + 感知占比
            n_ctx = feature.get('n_context', 1) or 1
            return [float(feature.get('perceptual_count', 0)),
                    float(feature.get('perceptual_count', 0)) / n_ctx]
        
        # 语义特征维度（默认模式 hybrid = 语义 + 2 感知维度）
        vec = [0.0] * (n_vocab + 2)
        for sememe, count in sememe_counts.items():
            if sememe in global_vocab:
                vec[global_vocab[sememe]] = count
        if mode == "hybrid":
            n_ctx = feature.get('n_context', 1) or 1
            vec[n_vocab] = feature.get('perceptual_count', 0)
            vec[n_vocab + 1] = feature.get('perceptual_count', 0) / n_ctx
        return vec
    
    def build_global_vocab(self, all_features: list[dict]) -> dict[str, int]:
        """从所有特征中构建全局义原词表。"""
        all_sememes = set()
        for f in all_features:
            all_sememes.update(f['sememes'].keys())
        return {s: i for i, s in enumerate(sorted(all_sememes))}


# ═══════════════════════════════════════════════════════════
# 模块 3：聚类与评估
# ═══════════════════════════════════════════════════════════

class ClusterEvaluator:
    """对围合特征做聚类，与金标对比。"""
    
    @staticmethod
    def evaluate(features: list[dict], gold_labels: list[str],
                 n_clusters: int | None = None,
                 mode: str = "hybrid") -> dict:
        """聚类评估。
        
        Args:
            features: extract_features 的输出列表
            gold_labels: 对应的金标列表（义项名称）
            n_clusters: 聚类数，默认取金标类别数
            mode: 'hybrid'/'perceptual_only'/'sememe_only'
        """
        if n_clusters is None:
            n_clusters = len(set(gold_labels))
        
        extractor = EnclosureFeatureExtractor()
        vocab = extractor.build_global_vocab(features)
        X = np.array([extractor.to_vector(f, vocab, mode=mode) for f in features])
        
        # 过滤全零行（没有义原的样本）
        valid_mask = X.sum(axis=1) > 0
        if valid_mask.sum() < 2:
            return {'error': 'too few valid samples', 'n_valid': int(valid_mask.sum())}
        
        X_valid = X[valid_mask]
        labels_valid = [gold_labels[i] for i in range(len(gold_labels)) if valid_mask[i]]
        
        # KMeans 聚类
        km = KMeans(n_clusters=n_clusters, random_state=CLUSTER_SEED, n_init=10)
        pred = km.fit_predict(X_valid)
        
        # 评估指标
        ari = adjusted_rand_score(labels_valid, pred)
        ami = adjusted_mutual_info_score(labels_valid, pred)
        
        # 随机基线（多次随机分配取均值）
        n_trials = 100
        rand_aris = []
        for seed in range(n_trials):
            rng = np.random.RandomState(seed)
            rand_pred = rng.randint(0, n_clusters, len(labels_valid))
            rand_aris.append(adjusted_rand_score(labels_valid, rand_pred))
        
        return {
            'n_samples': len(features),
            'n_valid': int(valid_mask.sum()),
            'n_clusters': n_clusters,
            'n_sememes_in_vocab': len(vocab),
            'ari': round(float(ari), 4),
            'ami': round(float(ami), 4),
            'rand_baseline_mean': round(float(np.mean(rand_aris)), 4),
            'rand_baseline_std': round(float(np.std(rand_aris)), 4),
            'passed': ari > np.mean(rand_aris) + 2 * np.std(rand_aris),
        }


# ═══════════════════════════════════════════════════════════
# 模块 4：数据加载
# ═══════════════════════════════════════════════════════════

def load_p0_data(split: str = "dev") -> list[dict]:
    """加载 P0-R 标准评测数据。
    
    Args:
        split: 'dev' 或 'blind'
    
    Returns:
        [{'id', 'sentence', 'target_word', 'target_occurrence', 'gold_sense'}]
    """
    suffix = "blind" if split == "blind" else "dev"
    input_path = os.path.join(P0_DATA_DIR, f"{suffix}_input.json")
    gold_path = os.path.join(P0_DATA_DIR, f"{suffix}_gold.json")
    
    with open(input_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        golds = json.load(f)
    
    # 合并
    gold_map = {}
    for g in golds:
        gold_map[g['id']] = g['_gold_label']
    
    result = []
    for item in inputs:
        result.append({
            'id': item['id'],
            'sentence': item['sentence'],
            'target_word': item['target_word'],
            'target_occurrence': item.get('target_occurrence', 1),
            'gold_sense': gold_map.get(item['id'], 'UNKNOWN'),
        })
    return result


def load_llm_data() -> list[dict]:
    """加载 LLM 生成的语境语料。
    
    格式：{word: {sense: [sentence1, sentence2, ...]}}
    每个词的每个义项有多句，可直接作为"不同语境"的样本。
    """
    with open(LLM_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    result = []
    for word, senses in data.items():
        for sense, sentences in senses.items():
            for sent in sentences:
                result.append({
                    'sentence': sent,
                    'target_word': word,
                    'gold_sense': sense,
                    'target_occurrence': 1,
                })
    return result


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def run_experiment(data_source: str = "dev", max_per_word: int | None = None):
    """运行 H-SG-1 实验主流程。
    
    Args:
        data_source: 'dev' 或 'blind' 或 'llm'
        max_per_word: 每个词最多取多少句（调试用）
    """
    print(f"\n{'='*60}")
    print(f"H-SG-1 实验：围合特征聚类验证")
    print(f"数据来源: {data_source}")
    print(f"{'='*60}\n")
    
    # 1. 加载数据
    if data_source == "llm":
        samples = load_llm_data()
    else:
        samples = load_p0_data(data_source)
    print(f"加载 {len(samples)} 条样本")
    
    # 按词分组后，每组内打乱顺序
    by_word: dict[str, list] = {}
    for s in samples:
        by_word.setdefault(s['target_word'], []).append(s)
    
    # 打乱每个词的样本顺序
    import random
    rng = random.Random(42)
    for word in by_word:
        rng.shuffle(by_word[word])
    
    # 2. 逐词提取围合特征
    extractor = EnclosureFeatureExtractor()
    word_results = {}
    
    for word, word_samples in sorted(by_word.items()):
        if max_per_word:
            word_samples = word_samples[:max_per_word]
        
        print(f"\n  [{word}] {len(word_samples)} 句...", end=" ", flush=True)
        
        features = []
        golds = []
        for s in word_samples:
            feat = extractor.extract_features(
                sentence=s['sentence'],
                target_word=s['target_word'],
                occurrence=s['target_occurrence'],
            )
            if feat['status'] == 'ok' and feat['n_sememes'] > 0:
                features.append(feat)
                golds.append(s['gold_sense'])
        
        print(f"有效 {len(features)}/{len(word_samples)} 句", end="")
        
        if len(features) < 3:
            print(" ❌ 样本太少，跳过")
            continue
        
        # 3. 聚类评估
        evaluator = ClusterEvaluator()
        result = evaluator.evaluate(
            features, golds,
            n_clusters=len(set(golds)),
        )
        
        word_results[word] = result
        
        if 'error' in result:
            print(f" ❌ {result['error']}")
        else:
            passed = "✅" if result['passed'] else "❌"
            print(f"  ARI={result['ari']:.4f} "
                  f"(随机基线={result['rand_baseline_mean']:.4f}±{result['rand_baseline_std']:.4f}) "
                  f"{passed}")
    
    # 4. 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    passed_count = sum(1 for r in word_results.values() if r.get('passed', False))
    total = len(word_results)
    print(f"通过: {passed_count}/{total}")
    for word, r in sorted(word_results.items()):
        if 'error' in r:
            print(f"  {word}: ❌ {r['error']}")
        else:
            mark = "✅" if r['passed'] else "❌"
            ari = r['ari']
            base = r['rand_baseline_mean']
            print(f"  {word}: ARI={ari:.4f} (基线={base:.4f}) {mark}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="H-SG-1 围合特征聚类实验")
    parser.add_argument("--data", choices=["dev", "blind", "llm"], default="dev",
                        help="数据来源")
    parser.add_argument("--max", type=int, default=None,
                        help="每个词最多取多少句")
    args = parser.parse_args()
    
    run_experiment(data_source=args.data, max_per_word=args.max)
