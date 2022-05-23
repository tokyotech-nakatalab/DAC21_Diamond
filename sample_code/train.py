from QuestionRecommender.datasets.dataset import AnsPredDataset
from QuestionRecommender.datasets.utils import ShelveDict
from QuestionRecommender.models.model import VariableModel
from QuestionRecommender.utils.trainer import SimpleTrainer
import torch
from torch import nn
import pickle
from loguru import logger
import json

from setting import *

def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)


question_answer_df = loadPickle(input_path + "question_answer_df.pkl")
categorical_question_features = loadPickle(input_path + "categorical_question_features.pkl")
numerical_question_features = loadPickle(input_path + "numerical_question_features.pkl")
categorical_user_features = loadPickle(input_path + "categorical_user_features.pkl")
numerical_user_features = loadPickle(input_path + "numerical_user_features.pkl")
qas_nc_add_to_user_dict = loadPickle(input_path + "qas_nc_add_to_user_dict.pkl")
OrdinalEncoder_question = loadPickle(input_path + "OrdinalEncoder_question.pkl")
OrdinalEncoder_users = loadPickle(input_path + "OrdinalEncoder_users.pkl")

logger.debug("data was loaded")


# hyperparameter

num_c = 8
num_g = 8
num_p = 48
num_u = int(OrdinalEncoder_users.get_params()["mapping"][0]["mapping"].max() + 1)
num_c = int(OrdinalEncoder_question.get_params()["mapping"][0]["mapping"].max() + 1)
num_hist_feature = 24

for v in numerical_question_features.values():
    sentence_embedding_dim = len(v["word2vec"])
    break

numerical_question_features = ShelveDict.from_dict(numerical_question_features)

question_features = {}
answer_features = {}

# Model Input
if "category_id" in use_features:
    question_features["category_id"] = {"num_id": num_c, "dim_out": dim_category}

if "hour_and_weekday" in use_features:
    question_features["hour_and_weekday"] = {
        "dim_hour": dim_hour,
        "dim_weekday": dim_week,
        "dim_out": dim_hour+dim_week,
    }

if "word2vec" in use_features:
    question_features["word2vec"] = {"dim_out": sentence_embedding_dim}

if "word_count" in use_features:
    question_features["word_count"] = {"dim_out": 1}

if "user_id" in use_features:
    question_features["user_id"] = {"num_user": num_u, "dim_out": dim_user}
    answer_features["user_id"] = {"num_user": num_u, "dim_out": dim_user}

if "generation_id" in use_features:
    question_features["generation_id"] = {"num_id": num_g, "dim_out": dim_gen}
    answer_features["generation_id"] = {"num_id": num_g, "dim_out": dim_gen}

if "prefecture_id" in use_features:
    question_features["prefecture_id"] = {"num_id": num_p, "dim_out": dim_pre}
    answer_features["prefecture_id"] = {"num_id": num_p, "dim_out": dim_pre}

if "register_days" in use_features:
    question_features["register_days"] = {"dim_out": 1}
    answer_features["register_days"] = {"dim_out": 1}

if "child_info" in use_features:
    question_features["child_info"] = {
        "dim_sex": dim_sex,
        "dim_b_month": dim_b_month,
        "dim_out": dim_sex+dim_b_month,
    }
    answer_features["child_info"] = {
        "dim_sex": dim_sex,
        "dim_b_month": dim_b_month,
        "dim_out": dim_sex+dim_b_month,
    }

if "b_year" in use_features:
    question_features["b_year"] = {"dim_out": 1}
    answer_features["b_year"] = {"dim_out": 1}
    
if "to_user_id" in use_features:
    answer_features["to_user_id"] = {
        "shareweights": True,
        "num_user": num_u,  
        "dim_out": dim_user,
        "num_head": num_head_touser,
        "dff": dim_user,
        "dropout": dropout,
        "n_layer": n_layer_touser,
    }

if "hist_features" in use_features:
    answer_features["hist_features"] = {"dim_out": num_hist_feature}

if "avg_hist_q_embs" in use_features:
    answer_features["avg_hist_q_embs"] = {"dim_out": sentence_embedding_dim}


last_features = {
    "dim_out": dim_out,
    "layer_length": layer_length
}

model_dict = {"question_features":question_features,"answer_features":answer_features,"last_features":last_features}
with open(output_path+model_name+"_info.json",'w') as f:
    json.dump(model_dict, f, indent=4)


dataset = AnsPredDataset(question_answer_df, categorical_question_features, numerical_question_features, categorical_user_features, numerical_user_features, qas_nc_add_to_user_dict, use_features, sentence_embedding_dim=sentence_embedding_dim)

device = "cuda"

model = VariableModel(question_features,answer_features,last_features)
#model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model.to(device)

optim = torch.optim.Adam(model.parameters())

loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

criterion =nn.BCEWithLogitsLoss(pos_weight=torch.ones(1)*(batch_size-1))
criterion.to(device)

trainer = SimpleTrainer(model,loader,criterion,optim,device=device)

torch.save(trainer.model.state_dict(), f'{output_path}{model_name}_epoch{0}')

logger.debug("training model")
for i in range(epochs):
    trainer.train(i)
    torch.save(trainer.model.state_dict(), f'{output_path}{model_name}_epoch{i+1}')
logger.debug("training has done!")
