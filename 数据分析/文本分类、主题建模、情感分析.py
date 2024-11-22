import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import jieba
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from sklearn.metrics import classification_report
# 数据读取
data = pd.read_csv('data.csv')

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

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(data_model['cus_comment'], data_model['target'], random_state=3, test_size=0.25)

# 停用词处理
with open("stopwords.txt", "r", encoding='utf-8') as infile:
    stopwords = [line.strip() for line in infile]

# 分词函数，现在只接受单个字符串
def segment(text):
    return ' '.join(jieba.cut(text))

# 将训练集的评论数据进行分词
x_train_seg = x_train.apply(lambda x: segment(x))

# TF-IDF转换
tv = TfidfVectorizer(stop_words=stopwords, max_features=3000, ngram_range=(1,2))
tv.fit(x_train_seg)

# 训练模型
classifier = MultinomialNB()
classifier.fit(tv.transform(x_train_seg), y_train)

# 模型评估
x_test_seg = x_test.apply(segment)
y_pred = classifier.predict(tv.transform(x_test_seg))
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", cm)

# 情感评分预测函数
def predict_sentiment(model, text):
    segmented = segment(text)  # 直接传入文本字符串
    return model.predict_proba(tv.transform([segmented]))[:,1][0]

# 测试样例
test1 = '因为晚上忙完就不早了，搜了附近很多店都关门了，于是就这家了!没想到意外惊喜，远超出预期!好吃，非常好吃，你看，很晚了店门口和店里还很多人，就说明门味错不了!'  # 5星好评
test2 = '糯米外皮不绵滑，豆沙馅粗躁，没有香甜味。12元一碗不值。'  # 1星差评

# 输出测试结果
print(f'好评实例的模型预测情感得分为 {predict_sentiment(classifier, test1)}')
print(f'差评实例的模型预测情感得分为 {predict_sentiment(classifier, test2)}')

# 计算准确率、召回率和F1分数
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 输出混淆矩阵和性能指标
print(f"准确率: {accuracy:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(data_model['cus_comment'], data_model['target'], random_state=3, test_size=0.25)

# 停用词处理
with open("stopwords.txt", "r", encoding='utf-8') as infile:
    stopwords = [line.strip() for line in infile]

# 分词函数
def segment(text):
    return ' '.join(jieba.cut(text))

# 将训练集的评论数据进行分词
x_train_seg = x_train.apply(lambda x: segment(x))

# TF-IDF转换
tv = TfidfVectorizer(stop_words=stopwords, max_features=3000, ngram_range=(1,2))
tv.fit(x_train_seg)

# 过采样：把0类样本复制10次，构造训练集
index_tmp = y_train == 0
y_tmp = y_train[index_tmp]
x_tmp = x_train_seg[index_tmp]
x_train2 = pd.concat([x_train_seg, pd.concat([x_tmp]*10)])
y_train2 = pd.concat([y_train, pd.concat([y_tmp]*10)])

# 使用过采样样本(简单复制)进行模型训练，并查看准确率
clf2 = MultinomialNB()
clf2.fit(tv.transform(x_train2), y_train2)

# 模型评估
y_pred2 = clf2.predict_proba(tv.transform(x_test.apply(segment)))[:,1]
print("ROC AUC Score:", roc_auc_score(y_test, y_pred2))

# 查看此时的混淆矩阵
y_predict2 = clf2.predict(tv.transform(x_test.apply(segment)))
cm2 = confusion_matrix(y_test, y_predict2)
print("混淆矩阵:\n", cm2)

# 测试函数
def ceshi(model, text):
    segmented = segment(text)
    return model.predict_proba(tv.transform([segmented]))[:,1][0]

# 输出测试结果
print("测试样例的模型预测情感得分为:", ceshi(clf2, '甜到怀疑人生!!!人生第一次喝水牛奶这么甜，好像糖不要钱一样，菠萝冰是菠萝块+水，一点冰都没有，也是没谁了，西米露也很难吃，大大的踩雷 ! !'))
# 将全部评论数据进行分词

# 数据读取
data = pd.read_csv('data.csv')

# 应用标签转换
data['target'] = data['stars'].map(convert_label)
data_model = data.dropna()

# 读取停用词文件
with open("stopwords.txt", "r", encoding='utf-8') as infile:
    stopwords = set([line.strip() for line in infile])

# 分词函数，现在会移除停用词
def segment(text):
    words = jieba.cut(text)
    return ' '.join(word for word in words if word not in stopwords)

# 将全部评论数据进行分词
x_all_seg = data_model['cus_comment'].apply(segment)

# 分词结果转换为列表
texts = [text.split() for text in x_all_seg]

# 创建词典
dictionary = corpora.Dictionary(texts)

# 创建词袋模型
corpus = [dictionary.doc2bow(text) for text in texts]

# 设置主题数
num_topics = 5

# 训练LDA模型
lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=0)

# 打印主题
topics = lda.print_topics(num_words=5)
for topic in topics:
    print(topic)

# 创建LDA模型的可视化
lda_vis = gensimvis.prepare(lda, corpus, dictionary)

# 显示可视化界面d
pyLDAvis.display(lda_vis)
# 保存LDA模型的可视化到HTML
pyLDAvis.save_html(lda_vis, "lda_visualization.html")

def convert_label(score):
    if score >= 4:
        return '非常正面'
    elif score == 3:
        return '正面'
    elif score == 2:
        return '中性'
    elif score == 1:
        return '负面'
    else:
        return '非常负面'

# 应用标签转换
data['target'] = data['stars'].apply(convert_label)

# 将标签转换为数字形式，为了使用sklearn的分类器
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 训练模型
classifier = MultinomialNB()
classifier.fit(tv.transform(x_train_seg), y_train_encoded)

# 模型评估
y_pred_encoded = classifier.predict(tv.transform(x_test_seg))

# 输出分类报告
# 将类别标签转换为字符串列表
target_names = [str(cls) for cls in label_encoder.classes_]
print(classification_report(y_test_encoded, y_pred_encoded, target_names=target_names))
