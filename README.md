# 大众点评评论文本挖掘


## 一、爬虫  

### 数据源获取  

爬取大众点评热门糖水店：“广州沙湾甜品食馆”前70页评论，爬取网页后从html页面中把需要的字段信息（顾客id、评论时间、评分、评论内容、口味、环境、服务、店铺ID）提取出来并存储到MYSQL数据库中。  

### 数据采集   

本次研究的数据采集主要是利用python的爬虫技术对大众点评网站中的糖水店铺“广州沙湾甜品食馆”的最近评论进行爬取，其中我使用了三个文件来实现网络爬虫功能，分别是：main.py，mysqls.py，CRAW_IP.py。其功能分别是：  
#### Mian.py:
**·主函数**：在 if __name__ == "__main__" 部分，定义了 craw_comment 函数作为主要执行逻辑，用于根据店铺ID和页码进行评论数据的爬取。  
**·爬取评论数据**：代码的主要目的是从大众点评中爬取店铺的评论数据。
**·请求设置**：它使用 requests 库发送HTTP请求，同时设置了伪造的用户代理（UserAgent）和cookies以模仿真实的浏览器访问。  
**·获取HTML文本**：getHTMLText 函数用于发送请求并获取网页的HTML内容。  
**·数据清洗与提取**：remove_emoji 函数用于从文本中移除emoji，因为MySQL数据库不支持长度为4个字符的emoji。parsePage 函数则从HTML中解析出评论数据，包括用户ID、评论时间、评分、口味、环境和服务等信息。  
**·存储数据**：解析出的数据通过调用 mysqls 模块的 save_data 函数保存到MySQL数据库中  
**·断点续传功能**：脚本还包含处理断点续传的逻辑，以便在爬虫意外中断时能从上次停止的地方继续。
#### Mysqls.py： 
**·连接MySQL数据库**：代码首先使用 pymysql 库来连接到本地的MySQL数据库。连接到名为 TESTDB 的数据库，并创建一个游标对象。  
**·创建数据表**：creat_table 函数用于创建一个名为 DZDP 的新数据表。表的结构包括多个字段，如 cus_id（顾客ID）、comment_time（评论时间）、comment_star（评分）、cus_comment（顾客评论）、kouwei（口味评分）、huanjing（环境评分）、fuwu（服务评分）和 shopID（店铺ID）。  
**·存储数据**：save_data 函数用于将爬取到的数据插入到 DZDP 数据表中。它接收一个包含这些字段信息的字典，并执行插入操作。  
**·错误处理和事务提交**：在数据插入过程中，代码通过 try-except 块来处理可能出现的异常，并在数据成功插入后提交事务。  
**·关闭数据库连接**：close_sql 函数用于关闭数据库连接。  
#### CRAW_IP.py
**·爬取代理IP**：文件中定义了一个名为 Proxies 的类。类的构造函数初始化了一个空的代理列表（self.proxies），然后调用 get_proxies 和 get_proxies_nn 方法来填充这个列表。这些方法访问 xicidaili.com 网站的不同页面，爬取代理IP地址和端口号。  
**·代理格式处理**：爬取的代理IP地址和端口号被格式化为 protocol://IP:port 的形式，并添加到代理列表中。  
**·验证代理IP**：verify_proxies 方法用于验证代理列表中的IP地址。它创建多个进程来并行验证每个代理，使用 requests.get 方法尝试通过代理访问 baidu.com。如果代理可以成功访问，它会被添加到新的队列中。  
**·输出有效代理**：有效的代理IP地址被保存到本地文件 proxies.txt 中  

在爬取完数据后，打开MySQL命令行，输入代码将爬取结果转为csv文件。
<div align="center">
  <img src=https://github.com/DontHeartMeGirl/images_for_README/blob/main/Mysql.png alt="登录界面图片">
</div>

## 二、文本数据预处理与探索性分析
### 数据预处理
 **去除非文本数据**：爬虫获取的数据非常多类似“\xa0”的非文本数据，而且都还有一些无意义的干扰数据，因此我们需要将文本中无关的数据清除。 
 
**中文分词与去除停用词**：使用jieba库把文本字符串处理为以空格区隔的分词字符串，以及列出所有需要排除的常见但不重要的词，在分词过程中过滤掉这些词。在统计词频之前，从每个评论的分词结果中移除这些停用词，再进行词频分析
 
 **改进后词频分析结果**：
<div align="center">
  <img src=https://github.com/DontHeartMeGirl/images_for_README/blob/main/images/%E7%82%B9%E8%AF%84.png alt="登录界面图片">
</div>
 
