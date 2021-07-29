"""Official evaluation script for KCT version 1.0.

Some code here has been copied from:
   https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
with some modifications.

To use this script, use command python evaluate.py <训练集> <预测结果>
"""
import argparse
import collections
import json
import re
import string
import sys
import pandas as pd
import numpy as np

OPTS = None


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for KAPT version 1.0.')
    parser.add_argument('--data_file', help='Input data file.')  # 数据文件，格式同训练集
    parser.add_argument('--pred_file', help='Model predictions.')  # CSV预测结果文件

    # if len(sys.argv) != 2:
    #     print('argument has error,' + str(len(sys.argv)) + ' not equal 2')
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def get_mean_scores(f1_scores):
    score_list = []
    for f1_score_item in f1_scores:
        f1_score_item = f1_score_item[0]
        score_list.append(f1_score_item.get("score"))
    mean_score = round(sum(score_list) / len(score_list), 5)
    return mean_score


def f1_score(df):
    f1_scores = []
    for item in df.iterrows():
        article = item[1]
        f1_scores_item = {}
        qid = article.get("qid")
        gold_answers = article.get("answer")
        pred_answers = json.loads(article.get("ret"))
        max_f1 = 0

        f1_scores_item["id"] = qid
        for gold in gold_answers:
            score = max(compute_f1(gold, a_pred) for a_pred in pred_answers)
            if score > max_f1:
                max_f1 = score
        f1_scores_item["score"] = max_f1

        f1_scores.append([f1_scores_item, gold_answers, pred_answers])
    f1_scores.sort(key=lambda x: x[0]['score'])
    # print(f1_scores)
    for fs in f1_scores:
        print(fs)
    score_mean = get_mean_scores(f1_scores)
    return score_mean


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def judge(standardResultFile, userCommitFile):
    """
	评估策略：计算F1得分
	standardResultFile  :  标准训练数据文件路径
	userCommitFile:  用户提交结果路径,为csv文件，包括id，ret两个字段
	"""
    err_type = '0'  # 错误类型
    err_code = None  # 错误码
    mean_f1 = None  # 求平均f1值
    try:
        label_df = pd.read_json(standardResultFile, lines=True)
        label_df.dropna(axis=0, inplace=True)
        pred_df = pd.read_csv(userCommitFile)
        pred_df.dropna(axis=0, inplace=True)
    except UnicodeError:
        err_type = '1'  # 编码错误
    except ValueError:
        err_type = '2'  # 文件格式错误
    else:

        match_flag = pred_df.shape[0] == label_df.shape[0]  # 样本记录数要一致
        cd_flag = len(set(np.unique(label_df['qid'])) - set(np.unique(pred_df['id'])))  # 检查id值域应一致

        chk_df = pd.merge(label_df, pred_df, left_on='qid', right_on='id', suffixes=('_label', '_pred'))
        if match_flag and cd_flag == 0:
            try:
                mean_f1 = f1_score(chk_df)
            except ZeroDivisionError:
                mean_f1 = 0
        else:
            err_type = '2'  # 记录数或值域不一致，预测结果存在空值

    err_code = None if err_type == '0' else err_type
    return [err_type, err_code, mean_f1]

class Evaluator:
    def evaluate(self, datafile, predfile):
        return judge(datafile, predfile)

if __name__ == '__main__':
    OPTS = parse_args()
    # 标准训练数据文件
    standardResultFile = OPTS.data_file
    # 选手预测结果
    userCommitFile = OPTS.pred_file

    result = judge(standardResultFile, userCommitFile)
    # print(result[0], end='')
    # print(';', end='')
    # print(result[1], end='')
    # print(';', end='')
    print("f1:", result[2], end='')
