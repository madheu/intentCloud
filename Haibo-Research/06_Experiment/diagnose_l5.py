"""诊断 L5：测试 HowNet 对语境词的覆盖"""
import OpenHowNet
hownet = OpenHowNet.HowNetDict()

test_words = ["手机", "芯片", "太阳", "发布", "新款", "屏幕", "应用", 
              "便宜", "采摘", "季节", "甜", "开花", "钱", "银行", "代码",
              "比", "了", "的", "是", "很", "这款", "这个"]

for w in test_words:
    data = hownet.get_sememes_by_word(w)
    ok = "YES" if data and any(s.get('sememes') for s in data) else "NO"
    count = len(data) if data else 0
    print(f"  {ok} {w}: {count} entries")
