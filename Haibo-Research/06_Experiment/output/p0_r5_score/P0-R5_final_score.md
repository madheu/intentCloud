# P0-R5 Final Score

## Summary

- Overall: **66/120 (55.00%)**
- Macro target accuracy: **57.41%**
- Macro sense recall: **55.12%**
- 95% Wilson CI: **[46.08%, 63.61%]**

## Target Accuracy

| Target | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| 光 | 10 | 12 | 83.33% |
| 口 | 5 | 12 | 41.67% |
| 头 | 6 | 18 | 33.33% |
| 小米 | 11 | 12 | 91.67% |
| 杜鹃 | 6 | 12 | 50.00% |
| 结 | 7 | 12 | 58.33% |
| 花 | 5 | 12 | 41.67% |
| 苹果 | 10 | 12 | 83.33% |
| 行 | 6 | 18 | 33.33% |

## Sense Recall

| Sense ID | Correct | Total | Recall |
| --- | --- | --- | --- |
| apple.company | 4 | 6 | 66.67% |
| apple.fruit | 6 | 6 | 100.00% |
| cuckoo.bird | 6 | 6 | 100.00% |
| cuckoo.flower | 0 | 6 | 0.00% |
| guang.figurative | 4 | 6 | 66.67% |
| guang.physical | 6 | 6 | 100.00% |
| hua.plant | 4 | 6 | 66.67% |
| hua.spend | 1 | 6 | 16.67% |
| jie.abstract | 1 | 6 | 16.67% |
| jie.concrete | 6 | 6 | 100.00% |
| kou.body | 0 | 6 | 0.00% |
| kou.space | 5 | 6 | 83.33% |
| tou.beginning | 5 | 5 | 100.00% |
| tou.body | 1 | 7 | 14.29% |
| tou.leader | 0 | 6 | 0.00% |
| xiaomi.company | 6 | 7 | 85.71% |
| xiaomi.grain | 5 | 5 | 100.00% |
| xing.approval | 0 | 6 | 0.00% |
| xing.industry | 0 | 5 | 0.00% |
| xing.row | 6 | 7 | 85.71% |

## Prediction Path Accuracy

| Path | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| MFS | 27 | 71 | 38.03% |
| L3 | 12 | 20 | 60.00% |
| L4 | 4 | 5 | 80.00% |
| L5 | 23 | 24 | 95.83% |

## Confusion Matrix

Rows are gold sense IDs; columns are predicted sense IDs.

| Gold \ Pred | apple.company | apple.fruit | cuckoo.bird | cuckoo.flower | guang.figurative | guang.physical | hua.plant | hua.spend | jie.abstract | jie.concrete | kou.body | kou.space | tou.beginning | tou.body | tou.leader | xiaomi.company | xiaomi.grain | xing.approval | xing.industry | xing.row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apple.company | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| apple.fruit | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cuckoo.bird | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cuckoo.flower | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| guang.figurative | 0 | 0 | 0 | 0 | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| guang.physical | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hua.plant | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hua.spend | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| jie.abstract | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| jie.concrete | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kou.body | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kou.space | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tou.beginning | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tou.body | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| tou.leader | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| xiaomi.company | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 1 | 0 | 0 | 0 |
| xiaomi.grain | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 |
| xing.approval | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| xing.industry | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| xing.row | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 6 |

## Errors

### p0r5-0001

- Sentence: 头痛的种类有超过二百种以上，有些是无害的，有些则会威胁生命。
- Target surface: 头
- Gold sense ID: `tou.body`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0004

- Sentence: 利用停产时间和生产任务不饱满的时间，抓紧对职工进行政治的和技术的培训，这是一种花费很小收益很大的智力开发。
- Target surface: 花
- Gold sense ID: `hua.spend`
- Predicted sense ID: `hua.plant`
- Prediction path: `MFS`

### p0r5-0005

- Sentence: 头领交厚，尝有书信往来。
- Target surface: 头
- Gold sense ID: `tou.leader`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0007

- Sentence: 在一些发展中国家，日常食物主要是以大米、小麦、玉米、小米为主，在发达国家，谷物消耗量较少，但仍然是食物中不可少的一部分。
- Target surface: 小米
- Gold sense ID: `xiaomi.company`
- Predicted sense ID: `xiaomi.grain`
- Prediction path: `L5`

### p0r5-0008

- Sentence: 新教徒处在反对国王和反对国教的双重斗争、以及需要对宗教和国家体制同时改革的危险困境中，因此他们在开头有点游移。
- Target surface: 头
- Gold sense ID: `tou.body`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0009

- Sentence: 呼吸系统感染：危重患者的口腔护理可降低呼吸机相关性肺炎的风险。
- Target surface: 口
- Gold sense ID: `kou.body`
- Predicted sense ID: `kou.space`
- Prediction path: `L3`

### p0r5-0010

- Sentence: 毛主席教导我们，各行各业都要学政治。
- Target surface: 行
- Gold sense ID: `xing.approval`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0015

- Sentence: 完善行业协会商会反腐倡廉制度，深化整治不正之风和腐败问题，一体推进不敢腐、不能腐、不想腐。
- Target surface: 行
- Gold sense ID: `xing.industry`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0020

- Sentence: 口腔变得疼痛，而且无故或容易出血。
- Target surface: 口
- Gold sense ID: `kou.body`
- Predicted sense ID: `kou.space`
- Prediction path: `L3`

### p0r5-0021

- Sentence: 鼓励开展产学研用联合攻关，吸引制造业龙头企业、跨国公司等在普陀设立研发中心。
- Target surface: 头
- Gold sense ID: `tou.leader`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0022

- Sentence: 杜鹃花科的模式属是欧石楠属。
- Target surface: 杜鹃
- Gold sense ID: `cuckoo.flower`
- Predicted sense ID: `cuckoo.bird`
- Prediction path: `MFS`

### p0r5-0023

- Sentence: 七、加强两地在跨境数据流动方面的交流，组成合作专责小组共同研究可行的政策措施安排。
- Target surface: 行
- Gold sense ID: `xing.approval`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0024

- Sentence: 70%的杜鹃花种类生长在海拔1700～3700米的地区。
- Target surface: 杜鹃
- Gold sense ID: `cuckoo.flower`
- Predicted sense ID: `cuckoo.bird`
- Prediction path: `MFS`

### p0r5-0031

- Sentence: 有研究显示，七成五的口腔癌个案与香烟有关系。
- Target surface: 口
- Gold sense ID: `kou.body`
- Predicted sense ID: `kou.space`
- Prediction path: `MFS`

### p0r5-0032

- Sentence: 第五层：颅骨膜（Pericranium）是头骨的骨膜，提供骨质的养分和修复的空间。
- Target surface: 头
- Gold sense ID: `tou.body`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0034

- Sentence: 二次文献具有明显的汇集性、系统性和可检索性，它的重要性在于使查找一次文献所花费的时间大大减少。
- Target surface: 花
- Gold sense ID: `hua.spend`
- Predicted sense ID: `hua.plant`
- Prediction path: `MFS`

### p0r5-0037

- Sentence: 通过控制变量产生不同的结果，实验有助于人们理解现象之间的因果关系。
- Target surface: 结
- Gold sense ID: `jie.abstract`
- Predicted sense ID: `jie.concrete`
- Prediction path: `MFS`

### p0r5-0038

- Sentence: 支持研究粤澳共同推进知识产权交易与融资，探讨粤澳两地合作开展知识产权评估互认等业务的可行性。
- Target surface: 行
- Gold sense ID: `xing.approval`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0039

- Sentence: 拿自己的模式作标准，去评判其他国家，把自己的模式强加于人，既不恰当，也根本行不通。
- Target surface: 行
- Gold sense ID: `xing.approval`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0040

- Sentence: 据长江日报反映：江西上饶某区第二十五村斗出一千多斤粮食，全部吃光，并且超过（估计必有贪污者）。
- Target surface: 光
- Gold sense ID: `guang.figurative`
- Predicted sense ID: `guang.physical`
- Prediction path: `MFS`

### p0r5-0045

- Sentence: 杜鹃花分布非常广泛，北半球大部分地方都有分布，南半球分布于东南亚和北澳大利亚。
- Target surface: 杜鹃
- Gold sense ID: `cuckoo.flower`
- Predicted sense ID: `cuckoo.bird`
- Prediction path: `MFS`

### p0r5-0047

- Sentence: 对孙小果、陈辉民、尚同军、黄鸿发等黑恶势力犯罪组织头目依法判处死刑，一批涉黑涉恶犯罪分子受到法律严惩。
- Target surface: 头
- Gold sense ID: `tou.leader`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0048

- Sentence: 在剥头皮时，就是移除这一层及较表层的组织。
- Target surface: 头
- Gold sense ID: `tou.body`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0050

- Sentence: 科学探究是迭代和非线性的：结论可能引发新的问题，而新证据可能迫使研究者修改甚至放弃原有的假说。
- Target surface: 结
- Gold sense ID: `jie.abstract`
- Predicted sense ID: `jie.concrete`
- Prediction path: `MFS`

### p0r5-0052

- Sentence: 日本学者川上泷弥最早在1910年记录丁香杜鹃分布于台湾。
- Target surface: 杜鹃
- Gold sense ID: `cuckoo.flower`
- Predicted sense ID: `cuckoo.bird`
- Prediction path: `MFS`

### p0r5-0053

- Sentence: 但有的国家对行业有分类规定。
- Target surface: 行
- Gold sense ID: `xing.industry`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0054

- Sentence: 促进工业经济平稳运行，加强原材料、关键零部件等供给保障，实施龙头企业保链稳链工程，维护产业链供应链安全稳定。
- Target surface: 头
- Gold sense ID: `tou.leader`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0058

- Sentence: 过去几十年中，以制造业为基础的行业将业务迁移到“发展中国家”经济体，其生产成本明显低于在“发达国家”经济体。
- Target surface: 行
- Gold sense ID: `xing.industry`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0059

- Sentence: 2004年总统选举花费了6.93亿美元，当时曾是历史最高。
- Target surface: 花
- Gold sense ID: `hua.spend`
- Predicted sense ID: `hua.plant`
- Prediction path: `MFS`

### p0r5-0061

- Sentence: 苹果公司和OpenAI并没有对此评价做出回应。
- Target surface: 苹果
- Gold sense ID: `apple.company`
- Predicted sense ID: `apple.fruit`
- Prediction path: `MFS`

### p0r5-0063

- Sentence: 12年来受买方一直被迫到外租房，因为购房将几代人积蓄都耗光，加上银行按揭货款压力巨大。
- Target surface: 光
- Gold sense ID: `guang.figurative`
- Predicted sense ID: `guang.physical`
- Prediction path: `MFS`

### p0r5-0064

- Sentence: 但后来发现原设想行不通，就多次修改计划。
- Target surface: 行
- Gold sense ID: `xing.approval`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0066

- Sentence: 一般来说，一个事件是很多原因综合产生的结果，而且原因都发生在较早时间点，而该事件又可以成为其他事件的原因。
- Target surface: 结
- Gold sense ID: `jie.abstract`
- Predicted sense ID: `jie.concrete`
- Prediction path: `MFS`

### p0r5-0069

- Sentence: 有些植物的花瓣退化了，或根本消失了。
- Target surface: 花
- Gold sense ID: `hua.plant`
- Predicted sense ID: `hua.spend`
- Prediction path: `L3`

### p0r5-0072

- Sentence: 2021年10月19日，苹果发布第三代AirPods。
- Target surface: 苹果
- Gold sense ID: `apple.company`
- Predicted sense ID: `apple.fruit`
- Prediction path: `MFS`

### p0r5-0073

- Sentence: 频繁的头痛会影响人际关系及工作。
- Target surface: 头
- Gold sense ID: `tou.body`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0074

- Sentence: 头骨的功能是支持并保护脑部和特殊感觉器官，并构成消化与呼吸系统的起始部，在一些动物中还起到辅助发声的作用。
- Target surface: 头
- Gold sense ID: `tou.body`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0077

- Sentence: 口腔通过嘴与外部世界相通，它不是一个无菌的地方。
- Target surface: 口
- Gold sense ID: `kou.body`
- Predicted sense ID: `kou.space`
- Prediction path: `L3`

### p0r5-0078

- Sentence: 大多数杜鹃花科植物有毒，如羊踯躅，但部分如蔓越莓、蓝莓、越橘等却是常见的可食用果物。
- Target surface: 杜鹃
- Gold sense ID: `cuckoo.flower`
- Predicted sense ID: `cuckoo.bird`
- Prediction path: `MFS`

### p0r5-0082

- Sentence: 针叶林下层有竹林，山区则有刺柏、红豆杉和杜鹃花。
- Target surface: 杜鹃
- Gold sense ID: `cuckoo.flower`
- Predicted sense ID: `cuckoo.bird`
- Prediction path: `MFS`

### p0r5-0087

- Sentence: 直接因素是直接影响结果的因素，也即无需任何介入因素（介入因素有时又称中介因素）。
- Target surface: 结
- Gold sense ID: `jie.abstract`
- Predicted sense ID: `jie.concrete`
- Prediction path: `MFS`

### p0r5-0091

- Sentence: 因为演出人员的片酬很高，加之某些场景花费巨大，主体拍摄通常是整个电影拍摄过程中耗资最多的环节。
- Target surface: 花
- Gold sense ID: `hua.spend`
- Predicted sense ID: `hua.plant`
- Prediction path: `MFS`

### p0r5-0096

- Sentence: 口腔健康与全身健康之间存在密切关联。
- Target surface: 口
- Gold sense ID: `kou.body`
- Predicted sense ID: `kou.space`
- Prediction path: `L3`

### p0r5-0097

- Sentence: 如今有三个好汉在那里扎寨：为头的唤做白衣秀士王伦，第二个唤做摸著天杜迁，第三个唤做云里金刚宋万。
- Target surface: 头
- Gold sense ID: `tou.leader`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0098

- Sentence: 很多公司会花费大笔资金在企业识别系统的建设方面，期望能设计出与众不同的识别，使它能更吸引目标群众。
- Target surface: 花
- Gold sense ID: `hua.spend`
- Predicted sense ID: `hua.plant`
- Prediction path: `MFS`

### p0r5-0099

- Sentence: 口腔内部温度恒定、湿度高，有许多狭窄的地方，是微生物生长的理想地方。
- Target surface: 口
- Gold sense ID: `kou.body`
- Predicted sense ID: `kou.space`
- Prediction path: `L3`

### p0r5-0100

- Sentence: 债务人的经营方案具有可行性。
- Target surface: 行
- Gold sense ID: `xing.approval`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0101

- Sentence: 其最顶层的叶是火红色、红色或白色的，因此经常被误会为花朵。
- Target surface: 花
- Gold sense ID: `hua.plant`
- Predicted sense ID: `hua.spend`
- Prediction path: `L3`

### p0r5-0107

- Sentence: 把各行各业都办成亦工亦农、亦文亦武的革命化的大学校，这是毛主席的一贯思想。
- Target surface: 行
- Gold sense ID: `xing.industry`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0110

- Sentence: 根据光照强度，洞穴可大致分为三个区域：洞口带、弱光带和黑暗带\[2]。
- Target surface: 口
- Gold sense ID: `kou.space`
- Predicted sense ID: `kou.body`
- Prediction path: `L3`

### p0r5-0115

- Sentence: 建立健全行业协会商会负责人提名、审核、公示、监督、退出等制度。
- Target surface: 行
- Gold sense ID: `xing.industry`
- Predicted sense ID: `xing.row`
- Prediction path: `MFS`

### p0r5-0116

- Sentence: 实验的目标各有不同，尺度也有大有小，但实验总是依赖可重复的操作，以及对结果的逻辑分析。
- Target surface: 结
- Gold sense ID: `jie.abstract`
- Predicted sense ID: `jie.concrete`
- Prediction path: `MFS`

### p0r5-0119

- Sentence: 事实与理由：洁柔公司成立于1999年，是国内领先的生活用纸巨头，是首家A股上市的生活用纸企业。
- Target surface: 头
- Gold sense ID: `tou.leader`
- Predicted sense ID: `tou.beginning`
- Prediction path: `MFS`

### p0r5-0120

- Sentence: 它会显示由一系列行与列构成的网格。
- Target surface: 行
- Gold sense ID: `xing.row`
- Predicted sense ID: `xing.approval`
- Prediction path: `L4`

## Data Integrity

- Prediction SHA-256: `34e57b7d1e1b3d133cfa3865fe8878c416c5154b9c556f79c21a27a3870fdf66` (matches frozen SHA)
- Gold records: 120; prediction records: 120.
- IDs were aligned by ID, not file order.
- Both files contain exactly `p0r5-0001` through `p0r5-0120`, with no missing, duplicate, or extra IDs.
- Every gold and predicted sense ID belongs to its record's candidate set.
- All Markdown gold labels parsed completely; no blank, pending, or illegal labels remain.
- Candidate inventory contains 20 sense IDs across 9 targets.

## Provenance

- The run used the runtime HowNet singleton wrapper.
- Prediction output SHA-256 is completely identical before and after wrapping.
- No predictions were rerun, no parameters were tuned, and no input data was modified during scoring.
