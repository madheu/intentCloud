"""极端简化的调试"""
NODES = ["苹果", "手机", "芯片"]
vocab = set(NODES)

sent = "苹果推出了搭载M4芯片的新款MacBook Pro"
words = [w for w in sent if w in vocab]
print(f"句子: {sent}")
print(f"匹配词: {words}")
print(f"数量: {len(words)}")

if len(words) >= 2:
    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            key = tuple(sorted([w1, w2]))
            print(f"  边: {key}")
