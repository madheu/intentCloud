"""调试 FROZEN_SEMEMES 为何全 0"""
import OpenHowNet
hownet = OpenHowNet.HowNetDict()

word = "苹果"
data = hownet.get_sememes_by_word(word)
print(f"get_sememes_by_word('{word}'): {len(data) if data else 0} entries")
if data:
    for i, entry in enumerate(data[:3]):
        print(f"  [{i}] keys={list(entry.keys())}")
        sememes = entry.get("sememes", [])
        print(f"      sememes类型: {type(sememes)}")
        if sememes:
            for s in sememes[:3]:
                print(f"        s={s!r} 类型={type(s)}")
                if isinstance(s, str):
                    print(f"        提取: {s.split('|')[0].strip()}")
