"""终极调试：文件内部定义 vs 外部传递"""
from collections import defaultdict

# 直接写死
vocab = {"苹果", "手机", "芯片"}
sent = "苹果推出了搭载M4芯片的新款MacBook Pro"

words = [w for w in sent if w in vocab]
print(f"直接测试: words={words}, len={len(words)}")

# 通过文件读入（模拟 p_resolve_08.py 的 NODES 加载方式）  
# 从 debug_tiny.py 自己读进来
with open(__file__, "r", encoding="utf-8") as f:
    content = f.read()
print(f"文件中'苹果'存在: {'苹果' in content}")
