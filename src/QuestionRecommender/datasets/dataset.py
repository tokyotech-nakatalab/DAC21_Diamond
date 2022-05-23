import shelve
from concurrent.futures import ThreadPoolExecutor as TPE
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from dateutil import parser
from loguru import logger
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .utils import loads

UserId = int
EncodedUserId = int


class AnsPredDataset(Dataset):
    ALL_FEATURES = [
        "category_id",
        "hour_and_weekday",
        "word2vec",
        "word_count",
        "user_id",
        "generation_id",
        "prefecture_id",
        "register_days",
        "child_info",
        "b_year",
        "to_user_id",
        "hist_features",
        "avg_hist_q_embs",
    ]

    def __init__(
        self,
        question_answer_df: pd.DataFrame,
        categorical_question_features: Dict[UserId, EncodedUserId],
        numerical_question_features: Dict,
        categorical_user_features: Dict[UserId, EncodedUserId],
        numerical_user_features: Dict,
        qas_nc_add_to_user_dict: Dict,
        feature_names: List[str],
        sentence_embedding_dim: int = -1,
    ) -> None:
        """質問回答者予測のためのデータセット

        Args:
            question_answer_df (pd.DataFrame): columns = ['question_id', 'question_user_id', 'answer_user_id', 'created']
            categorical_question_features (dict): (question_id, categorical_columns),
            numerical_question_features (dict): (question_id, numerical_columns),
            categorical_user_features (dict): (user_id, categorical_columns),
            numerical_user_features (dict): (user_id, numerical_columns),
            extra_question_features_directory (str): [description]
            extra_user_features_directory (str): [description]
        """
        self.df = question_answer_df
        self.categorical_question_features = categorical_question_features
        self._numerical_question_features = numerical_question_features
        self.categorical_user_features = categorical_user_features
        self.numerical_user_features = numerical_user_features
        self.qas_nc_add_to_user_dict = qas_nc_add_to_user_dict
        self.sentence_embedding_dim = sentence_embedding_dim

        # 各特徴量を使用するかのバイナリ
        self.use_category_id = "category_id" in feature_names
        self.use_hour_and_weekday = "hour_and_weekday" in feature_names
        self.use_word2vec = "word2vec" in feature_names
        self.use_word_count = "word_count" in feature_names
        self.use_user_id = "user_id" in feature_names
        self.use_generation_id = "generation_id" in feature_names
        self.use_prefecture_id = "prefecture_id" in feature_names
        self.use_register_days = "register_days" in feature_names
        self.use_child_info = "child_info" in feature_names
        self.use_b_year = "b_year" in feature_names
        self.use_to_user_id = "to_user_id" in feature_names
        self.use_hist_features = "hist_features" in feature_names
        self.use_avg_hist_q_embs = "avg_hist_q_embs" in feature_names

        for f_name in feature_names:
            if f_name not in self.ALL_FEATURES:
                print(f"{f_name}は実装されていません。タイプミスの可能性があります。")

        assert (not self.use_avg_hist_q_embs) or (self.sentence_embedding_dim > 0)

    def numerical_question_features(self, key):
        return self._numerical_question_features[str(int(key))]
        # return self._numerical_question_features[key]

    def __len__(self):
        """
        this method returns the total number of samples/nodes
        """
        return len(self.df)

    def get_question_features(self, question_id, question_user_id):
        question_dict = {}

        # カテゴリーIDを使うとき
        if self.use_category_id:
            category_id = [self.categorical_question_features[question_id]["category_id"]]
            question_dict["category_id"] = torch.LongTensor(category_id)

        # 時間と曜日を使うとき
        if self.use_hour_and_weekday:
            hour_and_weekday = [self.categorical_question_features[question_id][c] for c in ["hour", "weekday"]]
            question_dict["hour_and_weekday"] = torch.LongTensor(hour_and_weekday)

        # word2vecなどの文書埋め込みを使うとき
        if self.use_word2vec:
            word2vec = self.numerical_question_features(question_id)["word2vec"]
            # self.numerical_question_features(question_id)?
            question_dict["word2vec"] = torch.FloatTensor(word2vec)

        # 単語数を使うとき
        if self.use_word_count:
            word_count = [[self.numerical_question_features(question_id)["word_count"]]]
            question_dict["word_count"] = torch.FloatTensor(word_count)

        # ユーザーIDを使うとき
        if self.use_user_id:
            if question_user_id == -1:
                question_dict["user_id"] = torch.LongTensor([0])
            else:
                user_id = [self.categorical_user_features[question_user_id]["encoded_id"]]
                question_dict["user_id"] = torch.LongTensor(user_id)

        # ユーザーの世代IDを使うとき
        if self.use_generation_id:
            if question_user_id == -1:
                question_dict["generation_id"] = torch.LongTensor([0])
            else:
                generation_id = [self.categorical_user_features[question_user_id]["generation_id"]]
                question_dict["generation_id"] = torch.LongTensor(generation_id)

        # ユーザーの都道府県IDを使うとき
        if self.use_prefecture_id:
            if question_user_id == -1:
                question_dict["prefecture_id"] = torch.LongTensor([0])
            else:
                prefecture_id = [self.categorical_user_features[question_user_id]["prefecture_id"]]
                question_dict["prefecture_id"] = torch.LongTensor(prefecture_id)

        # ユーザーの登録日(user_id:1の登録日との差分)
        if self.use_register_days:
            if question_user_id == -1:
                question_dict["register_days"] = torch.FloatTensor([[0]])
            else:
                register_days = [[self.numerical_user_features[question_user_id]["register_days"]]]
                question_dict["register_days"] = torch.FloatTensor(register_days)

        # ユーザーの子供(末っ子)の性別、誕生月を使うとき
        if self.use_child_info:
            if question_user_id == -1:
                question_dict["child_info"] = torch.LongTensor([0, 0])
            else:
                child_info = [self.categorical_user_features[question_user_id][c] for c in ["sex", "b_month"]]
                question_dict["child_info"] = torch.LongTensor(child_info)

        # ユーザーの子供(末っ子)の誕生年を使うとき
        if self.use_b_year:
            if question_user_id == -1:
                question_dict["b_year"] = torch.FloatTensor([[0]])
            else:
                b_year = [[self.numerical_user_features[question_user_id]["b_year"]]]
                question_dict["b_year"] = torch.FloatTensor(b_year)

        return question_dict

    def get_answer_features(self, answer_user_id, created_at):
        answer_dict = {}

        # ユーザーIDを使うとき
        if self.use_user_id:
            if answer_user_id == -1:
                answer_dict["user_id"] = torch.LongTensor([0])
            elif answer_user_id not in self.categorical_user_features:
                answer_dict["user_id"] = torch.LongTensor([1])
            else:
                user_id = [self.categorical_user_features[answer_user_id]["encoded_id"]]
                answer_dict["user_id"] = torch.LongTensor(user_id)

        # ユーザーの世代IDを使うとき
        if self.use_generation_id:
            if answer_user_id == -1:
                answer_dict["generation_id"] = torch.LongTensor([0])
            elif answer_user_id not in self.categorical_user_features:
                answer_dict["generation_id"] = torch.LongTensor([0])
            else:
                generation_id = [self.categorical_user_features[answer_user_id]["generation_id"]]
                answer_dict["generation_id"] = torch.LongTensor(generation_id)

        # ユーザーの都道府県IDを使うとき
        if self.use_prefecture_id:
            if answer_user_id == -1:
                answer_dict["prefecture_id"] = torch.LongTensor([0])
            elif answer_user_id not in self.categorical_user_features:
                answer_dict["prefecture_id"] = torch.LongTensor([0])
            else:
                prefecture_id = [self.categorical_user_features[answer_user_id]["prefecture_id"]]
                answer_dict["prefecture_id"] = torch.LongTensor(prefecture_id)

        # ユーザーの登録日(user_id:1の登録日との差分)
        if self.use_register_days:
            if answer_user_id == -1:
                answer_dict["register_days"] = torch.FloatTensor([[0]])
            elif answer_user_id not in self.numerical_user_features:
                answer_dict["register_days"] = torch.FloatTensor([[0]])
            else:
                register_days = [[self.numerical_user_features[answer_user_id]["register_days"]]]
                answer_dict["register_days"] = torch.FloatTensor(register_days)

        # ユーザーの子供(末っ子)の性別、誕生月を使うとき
        if self.use_child_info:
            if answer_user_id == -1:
                answer_dict["child_info"] = torch.LongTensor([0, 0])
            elif answer_user_id not in self.categorical_user_features:
                answer_dict["child_info"] = torch.LongTensor([0, 0])
            else:
                child_info = [self.categorical_user_features[answer_user_id][c] for c in ["sex", "b_month"]]
                answer_dict["child_info"] = torch.LongTensor(child_info)

        # ユーザーの子供(末っ子)の誕生年を使うとき
        if self.use_b_year:
            if answer_user_id == -1:
                answer_dict["b_year"] = torch.FloatTensor([[0]])
            elif answer_user_id not in self.numerical_user_features:
                answer_dict["b_year"] = torch.FloatTensor([[0]])
            else:
                b_year = [[self.numerical_user_features[answer_user_id]["b_year"]]]
                answer_dict["b_year"] = torch.FloatTensor(b_year)

        user = None
        if self.use_to_user_id or self.use_hist_features:  # ユーザーの履歴情報とやり取りしたユーザーのどちらかを使うとき
            to_user_ids = [0] * 48  # ダミーで入れている。変更する場合は下で変更
            num_feats = [[0] * 24]  # ダミーで入れている。変更する場合は下で変更
            if answer_user_id in self.qas_nc_add_to_user_dict:
                user: pd.DataFrame = get_user(self.qas_nc_add_to_user_dict, answer_user_id)
                if len(user) != 0:
                    user_168h = get_user_168h(user, created_at)
                    if len(user_168h) != 0:
                        if self.use_to_user_id:
                            # やり取りしたユーザー
                            to_user_ids = make_to_user_ids(user_168h)

                            # 要調整
                            if len(to_user_ids) <= 48:
                                max_length = 48
                                to_user_ids += [0] * (max_length - len(to_user_ids))
                            else:
                                to_user_ids = to_user_ids[:48]

                            to_user_ids = torch.LongTensor(to_user_ids)

                        if self.use_hist_features:
                            # 行動履歴
                            user_168h_act_hist = make_user_168h_act_hist(user_168h)
                            user_act_hist = get_user_act_features(user_168h_act_hist, created_at)

                            num_feats = torch.FloatTensor(user_act_hist)

            if self.use_to_user_id:
                answer_dict["to_user_id"] = torch.LongTensor(to_user_ids)
            if self.use_hist_features:
                answer_dict["hist_features"] = torch.FloatTensor(num_feats)

        if self.use_avg_hist_q_embs:
            if user is None and answer_user_id in self.qas_nc_add_to_user_dict:
                user = get_user(self.qas_nc_add_to_user_dict, answer_user_id)
            if user is not None:
                parent_question_ids = get_user_last_k_question_ids(user, created_at)
                parent_question_embeddings = []
                for i in parent_question_ids:
                    try:
                        parent_question_embeddings.append(self.numerical_question_features(i)["word2vec"])
                    except:
                        logger.warning(f"Can't get the word2vec of parent_question_id={i}")
                if len(parent_question_embeddings) > 0:
                    parent_question_embeddings = sum(parent_question_embeddings) / len(parent_question_embeddings)
                else:
                    parent_question_embeddings = np.zeros(self.sentence_embedding_dim)
            else:
                parent_question_embeddings = np.zeros(self.sentence_embedding_dim)
            answer_dict["avg_hist_q_embs"] = torch.from_numpy(parent_question_embeddings).float()

        return answer_dict

    def __getitem__(self, idx: int) -> Tuple[Tuple[Dict[str, Tensor], Dict[str, Tensor]], Tensor]:
        """
        Generates one sample
        """

        datum = self.df.iloc[idx].to_dict()
        question_id = int(datum["question_id"])
        question_user_id = int(datum["question_user_id"])
        if np.isnan(datum["answer_user_id"]):
            answer_user_id = -1
            label = torch.BoolTensor([False])
        else:
            answer_user_id = int(datum["answer_user_id"])
            label = torch.BoolTensor([True])
        created_at = parser.parse(str(datum["created"]))

        with TPE() as exe:
            question_dict = exe.submit(self.get_question_features, question_id, question_user_id)
            answer_dict = exe.submit(self.get_answer_features, answer_user_id, created_at)
            question_dict = question_dict.result()
            answer_dict = answer_dict.result()

        # print(question_dict)
        # print(answer_dict)
        self._clear_cache()

        return (
            (question_dict, answer_dict),
            (label),
        )

    def _clear_cache(self):
        for name in vars(self).keys():
            m = getattr(self, name)
            if isinstance(m, shelve.Shelf):
                m.sync()


def get_user(qas_nc_add_to_user_dict, user_id):
    # 特定のユーザーのデータを抽出
    user = qas_nc_add_to_user_dict[user_id]
    return user


def get_user_168h(user, now):
    # 指定時刻から168時間以内のデータを抽出、未来のデータは含めない
    """
    df_qas_nc_168h = qas_nc_add_to_user.loc[
        (qas_nc_add_to_user.loc[:, "created"] >= now - timedelta(weeks=1)) & (qas_nc_add_to_user.loc[:, "created"] < now)
    ]
    """
    created = user["created"]
    since = now - timedelta(hours=168)
    left_index = np.searchsorted(created, since)
    right_index = np.searchsorted(created, now)
    if left_index < right_index:
        user_168h = user.iloc[left_index:right_index]
    else:
        user_168h = pd.DataFrame(columns=user.columns)
    return user_168h


def get_user_last_k_question_ids(user, now, k=3) -> List[int]:
    """
    userの直近k個回答のidsを返す
    """
    created = user["created"]
    idx = np.searchsorted(created, now)
    if idx > 0:
        user_last_k = user.iloc[:idx]
    else:
        user_last_k = pd.DataFrame(columns=user.columns)
    user_last_k = user_last_k[user_last_k["種類_回答"] > 0]
    return user_last_k.iloc[-k:]["parent_question_id"].tolist()


def make_user_168h_act_hist(user_168h):
    # 使わない列を削除
    user_168h = user_168h[["created", "種類_回答", "種類_検索", "種類_質問"]]

    # resampleで1時間ごとの合計行動数を算出
    user_168h_act = user_168h.resample("1h", on="created").sum()
    user_168h_act = user_168h_act.iloc[-168:].reset_index(drop=True)
    user_168h_act.index -= user_168h_act.index.max()  # 最大が0になるように
    user_168h_act["active"] = 1

    hours = list(range(-168, 1))
    user_168h_act_hist = user_168h_act.reindex(index=hours).fillna(0)
    return user_168h_act_hist


def get_user_act_features(user_168h_act_hist, now):
    user_act_hist = dict()

    def f(key, index, column):
        nonlocal user_act_hist
        user_act_hist[key] = user_168h_act_hist.loc[index, column]

    feat_cols = ["種類_回答", "種類_検索", "種類_質問", "active"]
    for t in [1, 24, 168]:
        for col in feat_cols:
            f(f"{col}_t-{t}", -t, col)
    # 24時間中の行動数の合計を特徴量に追加
    for dur in [24, 168]:
        user_168h_act_hist_sum = user_168h_act_hist.loc[-dur:, feat_cols].sum()
        for col in feat_cols:
            user_act_hist[f"answer_sum_{col}_{dur}h"] = user_168h_act_hist_sum[col]

    # 指定時刻の日付から得られる特徴量を追加
    user_act_hist["hour"] = int(now.strftime("%H"))
    user_act_hist["day"] = int(now.strftime("%d"))
    user_act_hist["month"] = int(now.strftime("%m"))
    user_act_hist["weekday"] = int(now.strftime("%w"))

    return [[user_act_hist[k] for k in user_act_hist]]


def make_to_user_ids(user_168h):
    # 一週間分のやり取りしたユーザーをリスト化
    to_user_ids = user_168h["to_encoded_id"].dropna().tolist()
    return to_user_ids


class DummyDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def get_question_features(self, index, u_index):
        q_feat = {
            "user_id": torch.LongTensor([index]),
            "generation": torch.LongTensor([1]),
            "region": torch.LongTensor([1]),
            "q_length": torch.FloatTensor([10]),
        }
        return q_feat

    def get_answer_features(self, user_id, time):
        a_feat = {
            "user_id": torch.LongTensor([user_id]),
            "generation": torch.LongTensor([1]),
            "region": torch.LongTensor([1]),
            "relation_hist": torch.LongTensor([1, 2, 3, 4]),
            "cum_feats": torch.FloatTensor([3, 4, 5]),
        }
        return a_feat

    def __getitem__(self, index):
        q_feat = self.get_question_features(index)
        a_feat = self.get_answer_features(index, 0)
        return q_feat, a_feat, torch.BoolTensor([True])
