import time
import argparse
import pickle
import random
import numpy as np
import torch
import logging
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# from BertForSearch import BertForSearch
from transformers import AdamW, get_linear_schedule_with_warmup, BertForMaskedLM, BertTokenizer
from point_dataset import BERTPretrainedMLMDataset
from tqdm import tqdm
import os, json, csv
from models import BertForKnowledgeQA
from evaluate import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--bert_model",type=str,default='./models/bert')

parser.add_argument("--config_file",type=str,help="The path to save log.",default='./conf.json')
parser.add_argument("--dataset_script_dir",type=str,help="The path to save log.",default='./data_scripts')
parser.add_argument("--dataset_cache_dir",type=str,help="The path to save log.",default='./cache')
parser.add_argument("--seed", default=0,type=int,help="seed")

parser.add_argument("--is_train",action="store_true",default=True)
parser.add_argument("--per_gpu_batch_size",default=32,type=int)
parser.add_argument("--learning_rate",default=5e-5,type=float,help="The initial learning rate for Adam.")
parser.add_argument("--train_file",type=str,default='./data/train_data.json')
parser.add_argument("--epochs",default=1,type=int,help="Total number of training epochs to perform.")

parser.add_argument("--is_dev",action="store_true",default=False)
parser.add_argument("--dev_file",type=str,default='/home/dou/knowledge_contest/newdata/processed_debug/inference_data_new.json')
parser.add_argument("--dev_src_file",type=str,default='/home/dou/knowledge_contest/newdata/dev.txt')

parser.add_argument("--is_test",action="store_true",default=True)
parser.add_argument("--test_file",type=str,default='./data/inference_data_1MASK.test.json')
parser.add_argument("--per_gpu_test_batch_size",default=80,type=int,help="The batch size.")
parser.add_argument("--mode", default=2,type=int,help="predict mode")
parser.add_argument("--save_path",default="./output",type=str,help="The path to save model.")

parser.add_argument("--log_path",default="/tmp/log.txt",type=str,help="The path to save log.")
parser.add_argument("--debug_eval",default=-1,type=int)


args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
logger = open(args.log_path, "a")
device = torch.device("cuda:0")
print(args)
logger.write("\n")

tokenizer = BertTokenizer.from_pretrained(args.bert_model)
train_data = args.train_file
test_data = args.test_file
dev_data = args.dev_file
try:
    os.makedirs(args.save_path)
except:
    print(f"{args.save_path} directory already exists!")
config = json.loads(open(args.config_file).read().strip())

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train_step(model, train_data, loss):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    
    model_output = model.forward(train_data)
    loss = model_output.loss
    return loss

def fit(model, X_train, X_test):
    train_dataset = BERTPretrainedMLMDataset(X_train, tokenizer, args.dataset_script_dir, args.dataset_cache_dir,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(t_total), num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    fct_loss = torch.nn.CrossEntropyLoss()
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        logger.flush()
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader)
        for i, training_data in enumerate(epoch_iterator):
            loss = train_step(model, training_data, fct_loss)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=args.learning_rate, loss=loss.item())

            # 每1/10个epoch跑一波dev集
            if args.debug_eval > 0:
                if args.is_dev and i > 0 and i % (one_epoch_step // args.debug_eval) == 0:
                # if i == 1:
                    predict(model, dev_data, args.mode, True)

            avg_loss += loss.item()
        if args.is_dev:
            predict(model, dev_data, args.mode, True)

        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
    

    
def train_model():
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model)
    model = BertForKnowledgeQA(bert_model)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    # 训练+dev验证
    fit(model, train_data, test_data)
    # 测试集，predict
    if args.is_test:
        predict(model, test_data, args.mode, False)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(args.save_path, "pytorch_model.bin"))
    logger.close()

def predict(model, X_test, mode=1, is_eval=False):
    model.eval()
    test_loss = []

    results = {}
    data = []
    with open(X_test) as ftest:
        for i, d in enumerate(ftest):
            data += [json.loads(d)]
    
    if mode == 1:
        # 只预测单mask情况
        idx = 0
        test_dataset = BERTPretrainedMLMDataset(X_test, tokenizer, args.dataset_script_dir, args.dataset_cache_dir,'test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
        with torch.no_grad():
            epoch_iterator = tqdm(test_dataloader, leave=False)
            # results = []
            for i, test_data in enumerate(epoch_iterator):
                with torch.no_grad():
                    for key in test_data.keys():
                        test_data[key] = test_data[key].to(device)
                model_output = model.forward(test_data) # bs, 2
                pred_logits = model_output.logits 
                logits_size = pred_logits.size()[0]
                for j in range(logits_size):
                    pred_logit = pred_logits[j]
                    masked_lm_positions = data[idx]['masked_lm_positions']
                    qid = data[idx]['qid']
                    if len(masked_lm_positions) > 1:
                        continue
                    mlm_position = masked_lm_positions[0]
                    pred_probs = torch.nn.functional.softmax(pred_logit[mlm_position], 0)
                    maxprobs, pred_idxs = torch.topk(pred_probs, 5)
                    pred_tokens = tokenizer.convert_ids_to_tokens(pred_idxs)
                    results[qid] = pred_tokens
                    idx += 1
                
    if is_eval:
        pred_score_path = os.path.join(args.save_path, 'score_dev.txt')
    else:
        pred_score_path = os.path.join(args.save_path, 'score.txt')
    dev_file_path = args.dev_src_file
    with open(pred_score_path, 'w') as swf:
        csvwriter = csv.writer(swf)
        csvwriter.writerow(["id","ret"])
        for qid in results:
            tokens_list= json.dumps(results[qid],ensure_ascii=False)
            csvwriter.writerow([qid, tokens_list])

    if is_eval:
    # 调用官方脚本进行验证，输入两个文件路径。
        eva = Evaluator()
        err_type, err_code, mean_f1 = eva.evaluate(dev_file_path, pred_score_path)
        print(f"mean_f1: {mean_f1}")
        logger.write(f"f1: {mean_f1}\n")
    logger.flush()

if __name__ == '__main__':
    set_seed(args.seed)
    if args.is_train:
        train_model()
