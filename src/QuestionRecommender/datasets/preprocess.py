from datetime import timedelta

import gensim
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
from janome.tokenizer import Tokenizer
from tqdm import tqdm

from .utils import dumps

# 前処理の関数はここにおいてください。


def merge_question_answer(dfq, dfa):
    dfq = dfq[["id", "user_id", "created"]]
    dfq = dfq.rename({"user_id": "question_user_id", "id": "question_id"}, axis=1)
    dfa = dfa[dfa["parent_answer_id"].isna()]
    dfa = dfa[["user_id", "question_id"]]
    dfa = dfa.rename({"user_id": "answer_user_id"}, axis=1)
    df = pd.merge(dfq, dfa, on="question_id", how="left")
    return df


def merge_user_children(dfu, dfc):
    dfu = dfu[["id", "generation_id", "prefecture_id", "created"]]
    dfu = dfu.rename({"id": "user_id"}, axis=1)
    dfu["created"] = pd.to_datetime(dfu["created"])
    # 登録日をuser_id:1との差分daysに変換
    dfu["register_days"] = dfu["created"] - dfu.loc[0, "created"]
    dfu["register_days"] = dfu["register_days"] // timedelta(days=1)
    dfu["register_days"] = dfu["register_days"].astype(int)
    dfc = dfc[["user_id", "birthday", "sex"]]
    dfc = dfc.sort_values(["user_id", "birthday"], ascending=[True, False])
    dfc["user_id_shift"] = dfc["user_id"].shift()
    dfc["youngest"] = 1
    # 末っ子以外はNanにする
    dfc.loc[dfc["user_id"] == dfc["user_id_shift"], "youngest"] = None
    dfc = dfc.dropna(subset=["youngest"])
    # 1600年生まれなどの例外処理
    dfc.loc[dfc["birthday"] < f"{pd.Timestamp.min:%Y%M%d}", "birthday"] = None
    dfc["birthday"] = pd.to_datetime(dfc["birthday"])
    dfc["b_year"] = dfc["birthday"].dt.year
    dfc["b_month"] = dfc["birthday"].dt.month
    dfc = dfc[["user_id", "sex", "b_year", "b_month"]]
    df = pd.merge(dfu, dfc, on="user_id", how="left")
    return df


def get_user_features(df_user):
    df_user = df_user[["id"]].copy()
    df_user = df_user.rename({"id": "user_id"}, axis=1)
    oe = OrdinalEncoder(cols="user_id", handle_missing="return_nan")
    oe.fit(df_user["user_id"])
    df_user["encoded_id"] = oe.transform(df_user["user_id"])
    user_dict = df_user.set_index("user_id").to_dict(orient="index")
    return user_dict, dict(), oe


def get_user_generation(df_user):
    df_user = df_user[["user_id", "generation_id", "prefecture_id", "register_days", "sex", "b_year", "b_month"]].copy()
    df_user["generation_id"] = df_user["generation_id"].fillna(0)
    df_user["prefecture_id"] = df_user["prefecture_id"].fillna(0)
    df_user["sex"] = df_user["sex"].fillna(0)
    df_user["b_year"] = df_user["b_year"].fillna(0)
    df_user["b_month"] = df_user["b_month"].fillna(0)
    oe = OrdinalEncoder(cols="user_id", handle_missing="return_nan")
    oe.fit(df_user["user_id"])
    df_user["encoded_id"] = oe.transform(df_user["user_id"])
    user_cat_dict = df_user[["user_id", "generation_id", "prefecture_id", "sex", "b_month", "encoded_id"]]
    user_num_dict = df_user[["user_id", "b_year", "register_days", "encoded_id"]]
    user_cat_dict = user_cat_dict.set_index("user_id").to_dict(orient="index")
    user_num_dict = user_num_dict.set_index("user_id").to_dict(orient="index")
    return user_cat_dict, user_num_dict, oe


def get_question_features(df_q):
    return dict(), dict()


def avg_feature_vector(sentence, model):
    size = model.vector_size

    t = Tokenizer()
    words = [token.base_form for token in t.tokenize(sentence)]
    feature_vec = np.zeros((size,), dtype="float32")  # 特徴ベクトルの入れ物を初期化
    for word in words:
        try:
            feature_vec = np.add(feature_vec, model.wv[word])
        except KeyError:
            continue
    if len(words) > 0:
        feature_vec = np.divide(feature_vec, len(words))
    return feature_vec


def get_question_categoty(df_q):
    df_q = df_q.rename({"id": "question_id", "vec": "word2vec"}, axis=1)
    df_q = df_q[["question_id", "category_id", "content", "created", "word2vec"]]
    df_q["content"] = df_q["content"].astype(str)
    df_q["word_count"] = df_q["content"].apply(lambda x: len(x))
    df_q["created"] = pd.to_datetime(df_q["created"], format="%Y-%m-%d %H:%M:%S")
    df_q["hour"] = df_q["created"].dt.hour
    df_q["weekday"] = df_q["created"].dt.strftime("%w").astype(int)
    oe = OrdinalEncoder(cols="category_id", handle_missing="return_nan")
    oe.fit(df_q["category_id"])
    df_q["category_id"] = oe.transform(df_q["category_id"])
    df_q = df_q.set_index("question_id")
    question_cat_dict = df_q[["category_id", "hour", "weekday"]].to_dict(orient="index")
    question_num_dict = df_q[["word_count", "word2vec"]].to_dict(orient="index")
    question_num_dict = {str(int(k)): v for k, v in question_num_dict.items()}
    return question_cat_dict, question_num_dict, oe


def get_qas_nocontent(df_qas):
    # content以外のdfにする。
    df_qas_nc = df_qas[["id", "user_id", "created", "種類"]].copy()
    df_qas_nc.loc[:, "created"] = pd.to_datetime(df_qas_nc.loc[:, "created"], format="%Y-%m-%d %H:%M:%S")
    # 行動のカテゴリを指定
    categories = ["回答", "検索", "質問"]
    # 種類列をカテゴリ型に変換
    df_qas_nc["種類"] = pd.Categorical(df_qas_nc["種類"], categories=categories)
    # 種類列の質問回答検索をダミー変数に変換
    df_qas_nc = pd.get_dummies(df_qas_nc, columns=["種類"])

    return df_qas_nc


def get_ans_nocontent(df_ans):
    # content以外を抽出
    df_ans_nc = df_ans[["id", "user_id", "question_id", "parent_answer_id", "is_best", "created", "種類"]].copy()
    return df_ans_nc


def get_to_userid(df_ans_nc, df_q):
    # 1. 親回答
    # dfから親回答のデータを抽出
    parent_answer = df_ans_nc[df_ans_nc["parent_answer_id"].isnull()]
    # questionの中のmergeに使うものだけ抽出
    df_q2 = df_q.copy()
    df_q2 = df_q2[["id", "user_id"]]
    # mergeするので名前を変更
    df_q2 = df_q2.rename({"user_id": "to_user_id", "id": "question_id"}, axis=1)
    # 親回答には質問者のuser_idをto_user_idにする
    pa_add_to_user = pd.merge(parent_answer, df_q2, on="question_id")

    # 2. 2番目以降の子回答
    # dfから子回答のデータを抽出
    child_answer = df_ans_nc.copy()
    child_answer = child_answer[child_answer["parent_answer_id"].notna()]
    # parent_anser_idとcreatedでソート
    child_answer = child_answer.sort_values(["parent_answer_id", "created"], ascending=[True, True])
    # 一つ前の子回答のuser_idをto_user_idにする
    child_answer["to_user_id"] = child_answer["user_id"].shift()
    # 最初の子回答は親回答のuser_idにしたいので一旦NaNにする
    child_answer.loc[child_answer["parent_answer_id"] != child_answer["parent_answer_id"].shift(), "to_user_id"] = None
    # 最初の子回答は削除
    ca_2nd_on_add_to_user = child_answer.copy()
    ca_2nd_on_add_to_user = ca_2nd_on_add_to_user.dropna()

    # 3. 1番目の子回答
    # 1番目の子回答だけ抽出する
    ca_1st = child_answer.copy()
    ca_1st = ca_1st[ca_1st["to_user_id"].isnull()]
    # to_user_id列を削除
    ca_1st = ca_1st.drop(["to_user_id"], axis=1)
    # mergeするので親回答のデータを名前変更
    pa_for_merge_with_ca_1st = parent_answer.copy()
    pa_for_merge_with_ca_1st = pa_for_merge_with_ca_1st[["id", "user_id"]]
    pa_for_merge_with_ca_1st = pa_for_merge_with_ca_1st.rename(
        {"id": "parent_answer_id", "user_id": "to_user_id"}, axis=1
    )
    # 名前変更した親回答のデータとmerge
    ca_1st_add_to_user = pd.merge(ca_1st, pa_for_merge_with_ca_1st, on="parent_answer_id")

    # to_userを加えたものを結合させる
    # 子回答のデータを結合
    ca_add_to_user = pd.concat([ca_1st_add_to_user, ca_2nd_on_add_to_user])
    # 親回答と子回答のデータを結合
    ans_add_to_user = pd.concat([pa_add_to_user, ca_add_to_user])

    # 必要な列だけ抽出、種類を追加
    ans_add_to_user = ans_add_to_user[["id", "user_id", "to_user_id", "種類", "question_id"]].copy()
    ans_add_to_user = ans_add_to_user.astype({"to_user_id": "int64"})
    ans_add_to_user["種類"] = 1
    ans_add_to_user = ans_add_to_user.rename({"種類": "種類_回答"}, axis=1)

    oe = OrdinalEncoder(cols="to_user_id", handle_missing="return_nan")
    oe.fit(ans_add_to_user["to_user_id"])
    ans_add_to_user["to_encoded_id"] = oe.transform(ans_add_to_user["to_user_id"])

    return ans_add_to_user


def merge_qas_ans_add_to_user(df_qas_nc, ans_add_to_user):
    ans_add_to_user = ans_add_to_user[["id", "to_encoded_id", "種類_回答", "question_id"]]
    ans_add_to_user = ans_add_to_user.rename(columns={"question_id": "parent_question_id"})

    qas_nc_add_to_user = pd.merge(df_qas_nc, ans_add_to_user, on=["id", "種類_回答"], how="left")
    """
        - id (int): 質問・回答・検索のid
        - to_encoded_id (EncodedUserId): 回答したときのparent_idのuserのid (labed encoded)
        - 種類_回答: 回答かどうかのバイナリ (i.e., is_answer) 
        - 種類_質問: 同上
        - 種類_検索: 同上
        - created: 行動時
        - user_id(UserId): 行動者id (not label encoded)
        - parent_question_id: 回答の質問id
    """

    return qas_nc_add_to_user


def to_dict_qas_ans_add_to_user(qas_nc_add_to_user):
    # userごとにリスト化
    qas_nc_add_to_user_dict = dict()
    qas_nc_add_to_user = qas_nc_add_to_user.groupby("user_id")[
        ["created", "種類_回答", "種類_検索", "種類_質問", "to_encoded_id", "parent_question_id"]
    ]
    for user_id, gf in tqdm(qas_nc_add_to_user):
        gf = gf.sort_values("created").reset_index(drop=True)
        gf = gf[(gf["種類_回答"] == 0) | (~gf.parent_question_id.isnull())]
        qas_nc_add_to_user_dict[user_id] = gf

    return qas_nc_add_to_user_dict
