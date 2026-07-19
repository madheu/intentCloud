"""调试 Unicode 匹配"""
NODES = ["苹果", "手机", "芯片"]
sent = "苹果推出了搭载M4芯片的新款MacBook Pro"

for w in NODES:
    print(f"'{w}' in sent: {w in sent}")
    print(f"  w bytes: {w.encode('utf-8').hex()}")
    
print(f"sent bytes start: {sent[:6].encode('utf-8').hex()}")

# Direct compare
a = "苹果"
b = sent[0:2]
print(f"a='{a}' b='{b}' a==b: {a == b}")
print(f"a hex: {a.encode('utf-8').hex()}")
print(f"b hex: {b.encode('utf-8').hex()}")
