"""调试 OpenHowNet API"""
import OpenHowNet

hownet = OpenHowNet.HowNetDict()

# 查看有哪些可用方法
methods = [m for m in dir(hownet) if not m.startswith('_')]
print("可用方法:", methods)

# 尝试不同方式获取数据
word = "苹果"
print(f"\n尝试获取 '{word}' 的数据：")

try:
    # 方法1: get
    result = hownet.get(word)
    print(f"get() 返回类型: {type(result)}")
    if result:
        print(f"  len={len(result)}" if hasattr(result, '__len__') else "")
        if isinstance(result, list):
            for i, item in enumerate(result):
                print(f"  [{i}] 类型={type(item).__name__}")
                for attr in dir(item):
                    if not attr.startswith('_'):
                        val = getattr(item, attr)
                        if val is not None:
                            val_str = str(val)[:100]
                            print(f"      .{attr} = {val_str}")
except Exception as e:
    print(f"get() 错误: {e}")

# 方法2: 检查是否有不同 API
for method_name in ['get_sememe', 'get_sememes', 'query', 'search', 'lookup', 'sense', 'get_sense']:
    if hasattr(hownet, method_name):
        try:
            method = getattr(hownet, method_name)
            result = method(word)
            print(f"\n{method_name}('{word}'): {type(result).__name__}")
            if result:
                print(f"  {str(result)[:200]}")
        except Exception as e:
            print(f"{method_name} 错误: {e}")
