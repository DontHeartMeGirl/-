import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 数据读取
data = pd.read_csv('../数据分析/data.csv')

# 标签转换函数
def convert_label(score):
    if score > 3:
        return 1
    elif score < 3:
        return 0
    else:
        return None

# 应用标签转换
data['target'] = data['stars'].map(convert_label)
data_model = data.dropna()

# 停用词处理
with open("../数据分析/stopwords.txt", "r", encoding='utf-8') as infile:
    stopwords = [line.strip() for line in infile]

# 分词函数
def segment(text):
    return ' '.join(jieba.cut(text))

# 将全部评论数据进行分词
x_all_seg = data_model['cus_comment'].apply(segment)

# TF-IDF转换
tv = TfidfVectorizer(stop_words=stopwords, max_features=3000, ngram_range=(1,2))
tfidf_vectors = tv.fit_transform(x_all_seg)

# 使用KMeans进行聚类
num_clusters = 5  # 假设我们选择5个聚类
km = KMeans(n_clusters=num_clusters, random_state=0)
km.fit(tfidf_vectors)

# 获取聚类结果
clusters = km.labels_

# 统计每个聚类的结果数量
cluster_counts = np.bincount(clusters)
for i in range(num_clusters):
    print(f"Cluster {i} 有 {cluster_counts[i]} 条评论")

