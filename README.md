# 🎓 大学生毕业就业样例数据集

> 📌 一份涵盖 10,000 条真实模拟记录的多功能数据集，专为 **数据挖掘**、**机器学习训练** 及 **统计分析** 场景打造。

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/Records-10%2C000-brightgreen?style=flat-square" alt="Records">
  <img src="https://img.shields.io/badge/Fields-8-orange?style=flat-square" alt="Fields">
  <img src="https://img.shields.io/badge/License-Open%20Data-lightgrey?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Format-CSV-purple?style=flat-square" alt="Format">
</p>

---

## 📊 数据集快照

| 指标 | 详情 |
| :--- | :--- |
| 📂 **样本总量** | 10,000 条 |
| 🏷️ **字段数量** | 8 个 |
| 🎯 **适用任务** | 分类 / 回归 / 聚类 / 关联分析 |
| 🧬 **数据类型** | 混合（分类特征 + 数值特征） |
| 🗂️ **文件格式** | CSV（UTF-8 编码） |
| 📅 **发布日期** | 2026-06-23 |

---

## 🧩 字段详细说明

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| 👤 **性别** | `分类` | 男 / 女 |
| 📚 **专业** | `分类` | 涵盖汉语言文学、计算机、金融学、法学、临床医学、环境工程、市场营销、电子信息、土木工程等 10+ 热门专业 |
| 🎖️ **学历** | `分类` | 本科 / 硕士 / 博士 |
| 🚀 **毕业去向** | `分类` | 就业 / 考研 / 公务员 / 出国 / 创业 / 自由职业 |
| 🏢 **就业行业** | `分类` | 互联网 / 政府机关 / 银行 / 医疗 / 教育 / 建筑 / 文化传媒 / 制造业等 |
| 📍 **工作城市** | `分类` | 北京 / 上海 / 深圳 / 杭州 / 成都 / 南京 / 武汉 / 西安 / 青岛等一线及新一线城市 |
| ⭐ **满意度评分** | `数值 (1-5)` | 1 = 非常不满意，5 = 非常满意 |
| 💼 **是否有实习经历** | `分类` | 是 / 否 |

---

## 🔍 数据特征概览

- ✅ **样本量充足**：10,000 条记录，满足深度学习与传统机器学习建模需求。
- ✅ **分布均衡**：文、理、工、商、医多学科交叉，去向与行业覆盖全面。
- ✅ **真实模拟**：符合当代大学生就业市场的基本逻辑与趋势。
- ✅ **无缺失值**：数据完整，无需额外处理缺失值，开箱即用。
- ✅ **标签丰富**：既包含分类标签（去向、行业），也包含连续标签（满意度），可玩性高。
- ✅ **城市集中**：聚焦一线及新一线城市，地域代表性较强。

---

## 🎯 适用场景与实战方向

### 🧠 分类预测
- 预测学生的 **毕业去向**（6 分类：就业/考研/公务员/出国/创业/自由职业）
- 预测 **就业行业** 或 **工作城市**
- 预测 **满意度等级**（可转化为高/低满意二分类）
- 预测学生 **是否有实习经历**

### 📈 回归分析
- 基于专业、学历、城市、实习经历等特征，构建模型预测 **满意度评分（1~5）**

### 🔗 关联规则与聚类
- 挖掘 **专业 → 行业** 的流向规律
- 分析 **学历与城市** 的偏好关系
- 利用 K-Means 等算法对学生进行 **人群画像分群**

### 🧹 特征工程练习
- 分类变量编码（One‑Hot Encoding / Label Encoding）
- 特征交叉与组合（如"学历 + 专业"联合特征）
- 数值标准化 / 归一化
- 特征重要性分析

### 📊 可视化报表
- 绘制就业去向饼图、行业分布柱状图
- 分析实习经历对满意度的提升效果
- 展示不同城市的平均满意度对比
- 各专业薪资/满意度趋势图（薪资可后续补充）

---

## 🚀 快速开始（代码示例）

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 加载数据
df = pd.read_csv('大学生毕业就业样例.csv', encoding='utf-8-sig')

# 2. 数据概览
print("📌 数据集基本信息：")
print(df.info())
print("\n📌 前5行预览：")
print(df.head())

# 3. 分类特征编码
categorical_cols = ['性别', '专业', '学历', '毕业去向', '就业行业', '工作城市', '是否有实习经历']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 4. 建模准备（以预测满意度为例 — 转化为二分类：满意/不满意）
df['满意度标签'] = (df['满意度评分'] >= 4).astype(int)

X = df.drop(['满意度评分', '满意度标签'], axis=1)
y = df['满意度标签']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 训练一个简单模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. 评估
y_pred = model.predict(X_test)
print(f"\n✅ 模型准确率: {accuracy_score(y_test, y_pred):.4f}")
```

---

## ⚠️ 注意事项

### 💡 数据性质
仅供教学、算法练习与学术研究使用。

### 🔍 字段解读
"就业行业"在"创业 / 出国 / 考研"等去向中依然存在，代表学生倾向或计划从事的行业，分析时请结合"毕业去向"字段联合理解，避免误判。

### 🔒 隐私安全
数据完全脱敏，不包含任何个人身份信息（如姓名、身份证号、联系方式），可公开安全使用。

### 🧪 拓展建议
您可以自行添加"薪资""绩点""学校层次"等字段，使数据集更加丰富，满足更多实战场景。

---

## 📌 版本记录

| 版本 | 更新内容 | 日期 |
| ---- | -------- | ---- |
| v1.0 | 初始版本发布，包含 10,000 条完整记录，8 个字段 | 2026-06-23 |

---

## 🤝 贡献与反馈

欢迎用于学习与教学！如有疑问或改进建议，可随时提出 Issue 或 Pull Request。

---

## 📄 许可证

本数据集采用 Open Data Commons Attribution License，允许自由使用、修改和分发，但需保留来源声明。

---

<p align="center"> <b>🌟 如果这个数据集对你有帮助，别忘了点个 Star 支持一下！</b> </p>
<p align="center"> <i>Happy Learning & Happy Coding! 🚀</i> </p>
