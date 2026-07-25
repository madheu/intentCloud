"""Debug: trace a single sentence through the enclosure pipeline."""
import sys, os, json, re
sys.path.insert(0, r"E:\intentCloud\Haibo-Research\06_Experiment")
from collections import Counter
import OpenHowNet

hownet = OpenHowNet.HowNetDict()
print("HowNet loaded")

# Test sentences from P0-R dev
sentences = [
    ("这个苹果很甜", "苹果", 1),
    ("华为苹果手机哪家好", "苹果", 1),
    ("这束花真漂亮", "花", 1),
    ("他花了很多钱", "花", 1),
]

def tokenize(sentence):
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', sentence)
    return tokens

def get_context(tokens, target_word, occurrence=1, window=3):
    count = 0
    target_pos = None
    for i, tok in enumerate(tokens):
        if target_word in tok or tok == target_word:
            count += 1
            if count == occurrence:
                target_pos = i
                break
    if target_pos is None:
        return None, tokens, None
    
    start = max(0, target_pos - window)
    end = min(len(tokens), target_pos + window + 1)
    context = []
    for i in range(start, end):
        if i != target_pos:
            context.append(tokens[i])
    return target_pos, tokens, context

print("\n=== 单句调试 ===")
for sent, target, occ in sentences:
    tokens = tokenize(sent)
    target_pos, _, context = get_context(tokens, target, occ)
    
    print(f"\n句子: {sent}")
    print(f"  目标: {target}")
    print(f"  tokens: {tokens}")
    print(f"  目标位置: {target_pos}")
    print(f"  上下文词: {context}")
    
    # Check HowNet for each context word
    for cw in context:
        sememes = hownet.get_sememes_by_word(cw, merge=True)
        if sememes:
            sememe_strs = [f"{s.zh}/{s.en}" for s in sememes[:5]]
            print(f"    {cw} → {len(sememes)}义原: {sememe_strs}")
        else:
            print(f"    {cw} → 无义原")

# Also try different tokenization approach
print("\n=== 尝试 jieba 分词（如果安装）===")
try:
    import jieba
    for sent, target, occ in sentences:
        tokens = list(jieba.cut(sent))
        _, _, context = get_context(tokens, target, occ)
        print(f"  {sent}: jieba={tokens}, context={context}")
except ImportError:
    print("  jieba not installed")
    # Install and try
    print("  Installing jieba...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "jieba", "-q"])
    import jieba
    for sent, target, occ in sentences:
        tokens = list(jieba.cut(sent))
        _, _, context = get_context(tokens, target, occ)
        print(f"  {sent}: jieba={tokens}, context={context}")
