"""探索 OpenHowNet：检查数据可用性 + 提取歧义词义原"""
import OpenHowNet
import os

# 检查数据路径
print("OpenHowNet 版本:", OpenHowNet.__version__ if hasattr(OpenHowNet, "__version__") else "?")

# 尝试初始化
try:
    hownet = OpenHowNet.HowNetDict()
    print("HowNet 初始化成功")
    print(f"  词数: {len(hownet)}" if hasattr(hownet, '__len__') else "  (无法获取总数)")
    
    # 检查具体词
    words = ["苹果", "纸", "光", "花", "行", "口", "手机", "太阳"]
    for w in words:
        try:
            entry = hownet.get(w)
            if entry:
                senses = entry if isinstance(entry, list) else [entry]
                print(f"\n{w} ({len(senses)} 个义项):")
                for i, s in enumerate(senses):
                    if hasattr(s, 'Def'):
                        print(f"  [{i}] DEF: {s.Def}")
                    if hasattr(s, 'sememes'):
                        print(f"      义原: {s.sememes}")
                    # 尝试不同属性名
                    for attr in ['definition', 'defination', 'sememe', 'concept', 'pos', 'example']:
                        if hasattr(s, attr):
                            val = getattr(s, attr)
                            if val:
                                print(f"      {attr}: {str(val)[:100]}")
            else:
                print(f"\n{w}: 未找到")
        except Exception as e:
            print(f"\n{w}: 错误 - {e}")
            
except Exception as e:
    print(f"HowNet 初始化失败: {e}")
    # 检查是否有数据文件
    for p in [os.path.expanduser("~"), os.getcwd(), "E:\\intentCloud\\models"]:
        files = []
        for root, dirs, fs in os.walk(p):
            for f in fs:
                if 'hownet' in f.lower() or 'sememe' in f.lower():
                    files.append(os.path.join(root, f))
        if files:
            print(f"  在 {p} 找到相关文件: {files[:5]}")
