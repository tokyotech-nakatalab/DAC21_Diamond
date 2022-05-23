import os
from QuestionRecommender.datasets.dataset import AnsPredDataset
from QuestionRecommender.datasets.utils import ShelveDict
from QuestionRecommender.models.model import VariableModel
from QuestionRecommender.utils.evaluate import OnTimeEvaluater, evaluate_from_vec
from QuestionRecommender.datasets.preprocess import merge_question_answer
import torch
import pickle
from loguru import logger
import json
from datetime import timedelta
import pandas as pd


from datetime import datetime, timedelta

dataset_path = "../dataset/"
testdata_path = "test/data/path"
model_path = "../model/"
model_name = "AllModel"
output_path = "../vectors/"

epoch = 0
time_begin = datetime(2021,7,1,0,0,0)
time_end = datetime(2021,7,2,0,0,0)
batch_size = 128
num_workers = 28
split_time_delta=timedelta(seconds=30)
target_time=timedelta(hours=24)
log_span=timedelta(minutes=60)


def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)


device = "cuda"

with open(model_path+model_name+"_info.json",'r') as f:
    model_info = json.load(f)
question_features = model_info["question_features"]
answer_features = model_info["answer_features"]

use_features = []
for k in question_features:
    if k not in use_features:
        use_features.append(k)
for k in answer_features:
    if k not in use_features:
        use_features.append(k)


last_features = model_info["last_features"]

model = VariableModel(question_features,answer_features,last_features)

#model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model.to(device)

model.load_state_dict(torch.load(f"{model_path}{model_name}_epoch{epoch}"))

logger.debug("model was loaded")

sentence_embedding_dim = question_features["word2vec"]["dim_out"]

question_answer_df = loadPickle(dataset_path + "question_answer_df.pkl")
categorical_question_features = loadPickle(dataset_path + "categorical_question_features.pkl")
numerical_question_features = loadPickle(dataset_path + "numerical_question_features.pkl")
categorical_user_features = loadPickle(dataset_path + "categorical_user_features.pkl")
numerical_user_features = loadPickle(dataset_path + "numerical_user_features.pkl")
numerical_question_features = ShelveDict.from_dict(numerical_question_features)
qas_nc_add_to_user_dict = loadPickle(dataset_path + "qas_nc_add_to_user_dict.pkl")
OrdinalEncoder_question = loadPickle(dataset_path + "OrdinalEncoder_question.pkl")
OrdinalEncoder_users = loadPickle(dataset_path + "OrdinalEncoder_users.pkl")

dataset = AnsPredDataset(question_answer_df, categorical_question_features, numerical_question_features, categorical_user_features, numerical_user_features, qas_nc_add_to_user_dict, use_features, sentence_embedding_dim=sentence_embedding_dim)

logger.debug("preprocess has done")



df_qas = pd.read_csv(f"{testdata_path}user_qas_test.csv")
df_a = pd.read_csv(f"{testdata_path}answer_test.csv")
df_q = pd.read_csv(f"{testdata_path}question_test.csv")

question_answer_df = merge_question_answer(df_q,df_a)

day = int(os.getenv('SGE_TASK_ID')) - 1
time_begin = time_begin + timedelta(days=day)
time_end = time_end + timedelta(days=day+2)

Evaluater = OnTimeEvaluater(dataset, model, question_answer_df, df_qas, device="cuda")
times,aucs,maps=Evaluater.evaluate(time_begin,time_end,batch_size,num_workers,split_time_delta,target_time,log_span,output_path,do_evaluate=False)

df = evaluate_from_vec(output_path, precision_k=5, use_gzip=True)
