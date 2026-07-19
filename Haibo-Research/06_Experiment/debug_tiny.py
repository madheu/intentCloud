"""绝对最小的边构建测试"""
from collections import defaultdict

NODES = ["苹果", "手机", "芯片"]
vocab = set(NODES)
SENTS = ["苹果推出了搭载M4芯片的新款MacBook Pro"]

edges = defaultdict(float)
for sent in SENTS:
    words = [w for w in sent if w in vocab]
    print(f"words={words}, len={len(words)}")
    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            key = tuple(sorted([w1, w2]))
            edges[key] += 1

print(f"edges={dict(edges)}, len={len(edges)}")
