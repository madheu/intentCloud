"""调试 HowNet 义原提取"""
import OpenHowNet
hownet = OpenHowNet.HowNetDict()

words = ["苹果", "纸", "光", "花", "行", "口", "太阳", "手机", "芯片", 
         "发布", "屏幕", "设计", "花园", "花钱", "钱", "银行", "代码",
         "好", "甜", "漂亮", "速度", "入口"]

for w in words:
    print(f"\n--- {w} ---")
    # get_sememes_by_word
    try:
        sememes = hownet.get_sememes_by_word(w)
        if sememes:
            print(f"  get_sememes_by_word: {len(sememes)} 义原")
            for s in sememes[:5]:
                print(f"    {str(s)[:80]}")
        else:
            print(f"  get_sememes_by_word: 空")
    except Exception as e:
        print(f"  get_sememes_by_word 错误: {e}")
    
    # get_sememe
    try:
        sememe_objs = hownet.get_sememe(w)
        if sememe_objs:
            print(f"  get_sememe: {len(sememe_objs)} 对象")
            for s in sememe_objs[:3]:
                print(f"    类型={type(s).__name__}")
                for attr in ['name', 'en_name', 'cn_name', 'definition']:
                    if hasattr(s, attr):
                        print(f"      .{attr} = {getattr(s, attr)}")
        else:
            print(f"  get_sememe: 空")
    except Exception as e:
        print(f"  get_sememe 错误: {e}")
    
    # get_sense - 查看义项
    try:
        senses = hownet.get_sense(w)
        if senses:
            print(f"  get_sense: {len(senses)} 义项")
            for s in senses[:3]:
                print(f"    {str(s)[:100]}")
        else:
            print(f"  get_sense: 空")
    except Exception as e:
        print(f"  get_sense 错误: {e}")
