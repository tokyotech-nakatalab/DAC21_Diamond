import zipfile
from datetime import datetime as dt

import pandas as pd
from loguru import logger
import pickle

from QuestionRecommender.datasets.preprocess import (
    get_ans_nocontent,
    get_qas_nocontent,
    get_question_categoty,
    get_to_userid,
    get_user_generation,
    merge_qas_ans_add_to_user,
    merge_question_answer,
    merge_user_children,
    to_dict_qas_ans_add_to_user,
)

# input folder path
DAC21data_path = "input/folder/path"
# output folder path
output_path = "../dataset/"

# data load
df_question_law = pd.read_pickle(DAC21data_path + "bert_cls.pkl")
df_answer_law = pd.read_csv(
    DAC21data_path + "answer_train.csv",
    usecols=["id", "user_id", "question_id", "parent_answer_id", "is_best", "created", "種類"],
)
df_user_law = pd.read_csv(DAC21data_path + "users.csv")
df_children_law = pd.read_csv(DAC21data_path + "children.csv")
qas_zip = zipfile.ZipFile(DAC21data_path + "user_qas.csv.zip")
df_qas = pd.read_csv(qas_zip.open("user_q&a.csv"), usecols=["id", "user_id", "created", "種類"])


# train_time
df_question_law["created"] = pd.to_datetime(df_question_law["created"], format="%Y-%m-%d %H:%M:%S")
df_question_law_train = df_question_law[df_question_law["created"] < dt(2021, 7, 1, 0, 0, 0)]

df_qas["created"] = pd.to_datetime(df_qas["created"], format="%Y-%m-%d %H:%M:%S")

logger.debug("datas was read")


question_answer_df = merge_question_answer(df_question_law_train, df_answer_law)
categorical_question_features, numerical_question_features, oe_q = get_question_categoty(df_question_law)
df_uc = merge_user_children(df_user_law, df_children_law)
categorical_user_features, numerical_user_features, oe_u = get_user_generation(df_uc)
df_qas_nc = get_qas_nocontent(df_qas)
df_ans_nc = get_ans_nocontent(df_answer_law)
ans_add_to_user = get_to_userid(df_ans_nc, df_question_law)
qas_nc_add_to_user = merge_qas_ans_add_to_user(df_qas_nc, ans_add_to_user)
qas_nc_add_to_user_dict = to_dict_qas_ans_add_to_user(qas_nc_add_to_user)

logger.debug("data was preprocessed")


def pickle_dump(obj, path):
    with open(path, mode="wb") as f:
        pickle.dump(obj, f)

pickle_dump(question_answer_df, output_path + "question_answer_df.pkl")
pickle_dump(categorical_question_features, output_path + "categorical_question_features.pkl")
pickle_dump(numerical_question_features, output_path + "numerical_question_features.pkl")
pickle_dump(categorical_user_features, output_path + "categorical_user_features.pkl")
pickle_dump(numerical_user_features, output_path + "numerical_user_features.pkl")
pickle_dump(qas_nc_add_to_user_dict, output_path + "qas_nc_add_to_user_dict.pkl")
pickle_dump(oe_q, output_path + "OrdinalEncoder_question.pkl")
pickle_dump(oe_u, output_path + "OrdinalEncoder_users.pkl")
logger.debug("data was saved")
