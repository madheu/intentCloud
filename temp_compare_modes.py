"""对比三种特征模式的 H-SG-1 结果：感知 vs 语义 vs 混合。"""
import sys, os
sys.path.insert(0, r"E:\intentCloud\Haibo-Research\06_Experiment")
from hsg1_enclosure_features import EnclosureFeatureExtractor, ClusterEvaluator, load_p0_data, load_llm_data

extractor = EnclosureFeatureExtractor()

# 试试 DEV 集（6词，自然语言）
print("=" * 70)
print("P0-R Dev 集：三模式对比")
print("=" * 70)
samples = load_p0_data("dev")
by_word = {}
for s in samples:
    by_word.setdefault(s['target_word'], []).append(s)

for word, word_samples in sorted(by_word.items()):
    features, golds = [], []
    for s in word_samples:
        feat = extractor.extract_features(s['sentence'], s['target_word'], 1)
        if feat['status'] == 'ok' and feat['n_sememes'] > 0:
            features.append(feat)
            golds.append(s['gold_sense'])
    if len(features) < 3:
        continue
    
    # 三种模式
    eval_hybrid = ClusterEvaluator.evaluate(features, golds, mode="hybrid")
    eval_percep = ClusterEvaluator.evaluate(features, golds, mode="perceptual_only")
    eval_sememe = ClusterEvaluator.evaluate(features, golds, mode="sememe_only")
    
    perc_avg = features[0].get('perceptual_count', 0)
    print(f"\n{word} ({len(features)}句, 感知词率~{perc_avg:.0f}/句):")
    print(f"  语义+感知: ARI={eval_hybrid.get('ari',0):.4f} (基={eval_hybrid.get('rand_baseline_mean',0):.4f})")
    print(f"  仅语义:    ARI={eval_sememe.get('ari',0):.4f} (基={eval_sememe.get('rand_baseline_mean',0):.4f})")
    print(f"  仅感知:    ARI={eval_percep.get('ari',0):.4f}")
