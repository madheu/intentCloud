"""Check if perceptual words appear in the dev set contexts."""
import sys, os, json
sys.path.insert(0, r"E:\intentCloud\Haibo-Research\06_Experiment")
from hsg1_enclosure_features import EnclosureFeatureExtractor, load_p0_data

extractor = EnclosureFeatureExtractor()
samples = load_p0_data("dev")

print("=== 各句子感知词出现情况 ===")
for s in samples:
    feat = extractor.extract_features(s['sentence'], s['target_word'], 1)
    perc_words = feat.get('perceptual_words', [])
    n_ctx = feat.get('n_context', 0)
    print(f"  [{s['gold_sense']}] {s['sentence'][:40]}")
    print(f"    上下文({n_ctx}): {feat['context_words']}")
    if perc_words:
        print(f"    感知词: {perc_words} ⭐")
    else:
        print(f"    感知词: (无)")
