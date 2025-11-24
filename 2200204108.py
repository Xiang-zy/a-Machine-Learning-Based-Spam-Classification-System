import os
import codecs
import re
import base64
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# 文本清洗
def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


# 提取邮件相关特征

# 发件人特征提取
def From_email(email):
    try:
        From_raw = re.search(r'From: (.*)', email).group(1)
    except:
        From_raw = ''
    From = ''
    name = re.search(r'=\?GB2312\?B\?(.*)\?=', From_raw, re.I)
    if name is None:
        name = ''
        From = From_raw
    else:
        name = name.group(1)
        try:
            name = base64.b64decode(name).decode('gb2312')
        except:
            try:
                name = base64.b64decode(name).decode('gbk')
            except:
                name = ''
        From = name + re.search(r'\?=(.*)', From_raw).group(1)
    return From


# 收件人特征提取
def To_email(email):
    To = re.search(r'^To: (.*)', email, re.M | re.I).group(1)
    return To


# 邮件主题特征提取
def Subject_email(email):
    Subject = re.search(r'=\?gb2312\?B\?(.*)\?=', email)
    if Subject is None:
        Subject = ''
    else:
        Subject = Subject.group(1)
        Subject = base64.b64decode(Subject)
        try:
            Subject = Subject.decode('gb2312')
        except:
            try:
                Subject = Subject.decode('gbk')
            except:
                Subject = ''
    return Subject


# 正文特征提取
def zhengwen_email(email):
    zhengwen = re.search(r'\n\n(.*)', email, re.S).group(1)
    zhengwen = clean_str(zhengwen)
    return zhengwen


# 特征连接
def combine_features(email):
    From = From_email(email)
    To = To_email(email)
    Subject = Subject_email(email)
    zhengwen = zhengwen_email(email)
    return " ".join([From, To, Subject, zhengwen])


# 构建索引文件函数
def Index_File():
    """index文件 路径--标签 对照表"""
    index_file = 'trec06c\\full\\index'
    f = codecs.open(index_file, 'r', 'gbk', errors='ignore')
    table = defaultdict(list)
    for line in f:
        label, path = line.strip().split()
        if label == 'spam':  # 是垃圾邮件
            label = 1
        else:
            label = 0
        table['label'].append(label)
        table['path'].append(path)
    table = pd.DataFrame(data=table)
    return table


# 构建数据集
def build_dataset():
    table = Index_File()
    path = 'trec06c\\data'
    dirs = os.listdir(path)
    emails = []
    labels = []
    for dir in dirs:
        dir_path = path + '\\' + dir
        files = os.listdir(dir_path)
        for file in files:
            file_path = dir_path + '\\' + file
            with codecs.open(file_path, 'r', 'gbk', errors='ignore') as f:
                email = f.read()
            combined_feature = combine_features(email)
            emails.append(combined_feature)
            flag = table[table['path'] == '../data/' + dir + '/' + file]['label'].values[0]
            labels.append(flag)
    return emails, labels


# 数据集划分
def split_dataset(emails, labels, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# 训练模型
def train_model(X_train, y_train, X_test=None, y_test=None, algorithm='naive_bayes'):
    if algorithm == 'naive_bayes':
        model = MultinomialNB()
    elif algorithm == 'svm':
        model = SVC()
    model.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

    return model


# 构建数据集
emails, labels = build_dataset()
# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = split_dataset(emails, labels)
# 特征提取
vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)
# 模型训练svm
model = train_model(X_train_features, y_train, X_test_features, y_test, algorithm='svm')
# 模型训练naive_bayes
model = train_model(X_train_features, y_train, X_test_features, y_test, algorithm='naive_bayes')
