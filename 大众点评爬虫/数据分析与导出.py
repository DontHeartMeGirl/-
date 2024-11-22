import pandas as pd
import pymysql
import jieba
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import seaborn as sns

# 设置中文字体，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统中的中文字体
plt.rcParams['axes.unicode_minus'] = False

# 数据库配置信息
config = {
    'host': "localhost",
    'user': "root",
    'password': "Cwh2432211491",
    'database': "testdb"
}

# 预处理函数：移除非中文字符
def preprocess_text(text):
    return re.sub(r'[^\u4e00-\u9fa5]', '', text)

def connect_db(config):
    try:
        engine = create_engine(f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}/{config['database']}")
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def read_data(engine, sql):
    try:
        data = pd.read_sql(sql, engine)
        return data
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

def clean_data(data):
    data.loc[data['comment_star'] == 'sml-str1', 'comment_star'] = 'sml-str10'
    data['stars'] = data['comment_star'].str.extract(r'(\d+)').astype(float) / 10
    data['cus_comment'] = data['cus_comment'].apply(preprocess_text)
    data['comment_len'] = data['cus_comment'].str.len()
    data['cus_comment'] = data['cus_comment'].str.replace('收起评价', '')
    return data

def clean_time_data(data):
    data = data.copy()  # 创建数据副本
    data['comment_time'] = data['comment_time'].astype(str)
    data['comment_time'] = data['comment_time'].str.extract(r'(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)', expand=False)
    data['comment_time'] = pd.to_datetime(data['comment_time'], errors='coerce')
    data = data[data['comment_time'].notna()]
    return data

def plot_data(data):
    data = clean_time_data(data)

    # 星级分布图
    sns.countplot(data=data, x='stars')
    plt.title('星级分布图')
    plt.show()

    # 时间特征提取
    data.loc[:, 'year'] = data['comment_time'].dt.year
    data.loc[:, 'month'] = data['comment_time'].dt.month
    data.loc[:, 'weekday'] = data['comment_time'].dt.weekday
    data.loc[:, 'hour'] = data['comment_time'].dt.hour

    # 各星期的小时评论数分布图
    fig, ax = plt.subplots(figsize=(14, 4))
    df = data.groupby(['hour', 'weekday']).count()['cus_id'].unstack()
    df.plot(ax=ax, style='-.')
    plt.title('各星期的小时评论数分布图')
    plt.show()

    # 评论长度分布图
    sns.boxplot(x='stars', y='comment_len', data=data)
    plt.ylim(0, 600)
    plt.title('评论长度分布图')
    plt.show()

def text_analysis(data):
    # 加载停用词
    with open('../大众点评爬虫/stopwords.txt', 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])

    # 分词并过滤停用词
    words = []
    for comment in data['cus_comment']:
        words.extend([word for word in jieba.cut(comment) if word not in stopwords])

    # 计算词频
    word_freq = Counter(words)
    top_words = word_freq.most_common(10)
    for word, freq in top_words:
        print(f"单词：{word}；出现频次：{freq}")

    # 生成词云，仅展示前十个词语
    word_freq_top_10 = word_freq.most_common(10)
    wordcloud = WordCloud(
        font_path='C:/Windows/Fonts/simhei.ttf',
        width=800,
        height=600,
        background_color='white'
    ).generate_from_frequencies(dict(word_freq_top_10))

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('评论词云（前十个词语）')
    plt.show()


def main():
    engine = connect_db(config)
    if engine:
        data = read_data(engine, "SELECT * FROM dzdp;")
        engine.dispose()
        if data is not None:
            data = clean_data(data)
            plot_data(data)
            text_analysis(data)
            data.to_csv('data3.csv', index=False)

if __name__ == "__main__":
    main()
