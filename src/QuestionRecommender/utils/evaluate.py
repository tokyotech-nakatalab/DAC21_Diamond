import glob
import gzip
import pickle
from concurrent.futures import ThreadPoolExecutor as TPE
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from black import json
from sklearn.metrics import auc, average_precision_score, ndcg_score, recall_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FeatureName = str


class Evaluater:
    def __init__(self, dataset, model, question_answer_df, df_users, device="cuda"):
        self.dataset = dataset
        self.model = model
        self.model.to(device)
        question_answer_df["created"] = pd.to_datetime(question_answer_df["created"], format="%Y-%m-%d %H:%M:%S")
        self.question_answer_df = question_answer_df
        self.user_list = df_users["id"].tolist()
        self.dict_user = {y: x for x, y in enumerate(self.user_list)}
        self.device = device

    def AUC(self, question_matrix, answer_matrix, labels):
        prediction = torch.matmul(question_matrix, torch.transpose(answer_matrix, 0, 1))
        prediction = prediction.reshape(-1).detach().numpy()
        labels = labels.reshape(-1).detach().numpy()
        fpr, tpr, thresholds = roc_curve(labels, prediction)
        return auc(fpr, tpr)

    def MAP(self, question_matrix, answer_matrix, labels):
        scores = torch.matmul(question_matrix, torch.transpose(answer_matrix, 0, 1))
        return average_precision_score(labels.detach().numpy(), scores.detach().numpy(), average="samples")

    def get_vectors(self, time, batch_size, num_workers):  # time=dt(Y,m,d,H)
        qdataset = QuestionDataset(self.dataset, self.question_answer_df, time, self.dict_user)
        qloader = DataLoader(qdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        q_vecs = []
        labels = []
        with tqdm(qloader) as pbar:
            for question_feature, label in pbar:
                for feature_name in question_feature:
                    question_feature[feature_name] = question_feature[feature_name].to(self.device, non_blocking=True)
                with torch.no_grad():
                    q_vec = self.model.question_encoder(question_feature)
                q_vecs.append(q_vec.to("cpu"))
                labels.append(label)
                pbar.set_description(f"making question_vector at {time}")
        q_vecs = torch.cat(q_vecs, dim=0)
        labels = torch.cat(labels, dim=0)

        adataset = AnswerDataset(self.dataset, self.user_list, time)
        aloader = DataLoader(adataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        a_vecs = []
        with tqdm(aloader) as pbar:
            for answer_feature in pbar:
                for feature_name in answer_feature:
                    answer_feature[feature_name] = answer_feature[feature_name].to(self.device, non_blocking=True)
                with torch.no_grad():
                    a_vec = self.model.answer_encoder(answer_feature)
                a_vecs.append(a_vec.to("cpu"))
                pbar.set_description(f"making answer_vector at {time}")
        a_vecs = torch.cat(a_vecs, dim=0)

        return q_vecs, a_vecs, labels

    def evaluate(self, time_begin, time_end, batch_size, num_workers, vec_save_folder=None):
        tmp_time = time_begin + timedelta(hours=1)
        self.times = []
        self.aucs = []
        self.maps = []
        while (tmp_time >= time_begin) and (tmp_time < time_end):
            self.times.append(tmp_time)
            qvec, uvec, qa_label = self.get_vectors(tmp_time, batch_size, num_workers)
            if not (vec_save_folder is None):
                vectors = {"question": qvec, "user": uvec, "label": qa_label}
                time_str = tmp_time.strftime("%Y-%m-%d_%H")
                with open(vec_save_folder + f"/{time_str}.pkl", "wb") as f:
                    pickle.dump(vectors, f)
            auc_score = self.AUC(qvec, uvec, qa_label)
            self.aucs.append(auc_score)
            map_score = self.MAP(qvec, uvec, qa_label)
            self.maps.append(map_score)

            tmp_time += timedelta(hours=1)
        return self.times, self.aucs, self.maps


class QuestionDataset(Dataset):
    def __init__(
        self,
        get_question_features,
        question_answer_df,
        time,
        user_dict,
        target_time=timedelta(hours=0),
    ) -> None:
        self.get_question_features = get_question_features
        self.time = time
        dt_b = time - target_time
        df_hour = question_answer_df[(question_answer_df["created"] >= dt_b) & (question_answer_df["created"] < time)]
        df_agg = df_hour.groupby(["question_id"]).agg(list).reset_index()
        df_agg["question_user_id"] = df_agg["question_user_id"].apply(lambda x: x[0])
        self.df_agg = df_agg
        self.user_dict = user_dict

    def __len__(self):
        """
        this method returns the total number of samples/nodes
        """
        return len(self.df_agg)

    def __getitem__(self, idx: int) -> Tuple[Dict[FeatureName, torch.Tensor], Dict[FeatureName, torch.Tensor]]:
        """
        Generates one sample
        """

        datum = self.df_agg.iloc[idx].to_dict()
        question_id = int(datum["question_id"])
        question_user_id = int(datum["question_user_id"])
        answer_user_ids = datum["answer_user_id"]  # list
        delta_seconds = (self.time - datum["created"][0]).seconds

        q_feature = self.get_question_features(question_id, question_user_id)

        label = torch.zeros(len(self.user_dict))
        for user_id in answer_user_ids:
            if np.isnan(user_id):
                break
            if user_id in self.user_dict:
                label[self.user_dict[user_id]] = 1  # ここあってるか要検証

        return q_feature, label, delta_seconds


class AnswerDataset(Dataset):
    def __init__(
        self,
        get_answer_features,
        user_list,
        time,
    ):
        self.get_answer_features = get_answer_features
        self.user_list = user_list
        self.time = time

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        user = self.user_list[index]
        a_feature = self.get_answer_features(user, self.time)
        return a_feature


class OnTimeEvaluater:
    def __init__(self, dataset, model, question_answer_df, df_qas, device="cuda"):
        self.dataset = dataset
        self.model = model
        self.model.to(device)
        question_answer_df["created"] = pd.to_datetime(question_answer_df["created"], format="%Y-%m-%d %H:%M:%S")
        self.question_answer_df = question_answer_df
        df_qas["created"] = pd.to_datetime(df_qas["created"], format="%Y-%m-%d %H:%M:%S")
        self.df_qas = df_qas
        self.device = device

    def AUC(self, question_matrix, answer_matrix, labels):
        prediction = torch.matmul(question_matrix, torch.transpose(answer_matrix, 0, 1))
        prediction = prediction.reshape(-1).detach().numpy()
        labels = labels.reshape(-1).detach().numpy()
        fpr, tpr, thresholds = roc_curve(labels, prediction)
        return auc(fpr, tpr)

    def MAP(self, question_matrix, answer_matrix, labels):
        scores = torch.matmul(question_matrix, torch.transpose(answer_matrix, 0, 1))
        return average_precision_score(labels.detach().numpy(), scores.detach().numpy(), average="samples")

    def get_vectors(self, time, batch_size, num_workers, target_time):  # time=dt(Y,m,d,H)
        qdataset = QuestionDataset(
            self.dataset.get_question_features, self.question_answer_df, time, self.dict_user, target_time
        )
        question_ids = qdataset.df_agg["question_id"]
        qloader = DataLoader(qdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        q_vecs = []
        labels = []
        delta_seconds = []
        with tqdm(qloader) as pbar:
            for question_feature, label, delta_m in pbar:
                for feature_name in question_feature:
                    question_feature[feature_name] = question_feature[feature_name].to(self.device, non_blocking=True)
                with torch.no_grad():
                    q_vec = self.model.question_encoder(question_feature)
                q_vecs.append(q_vec.to("cpu"))
                labels.append(label)
                delta_seconds.append(delta_m)
                pbar.set_description(f"making question_vector at {time}")
        if len(q_vecs) == 0:
            q_vecs = torch.FloatTensor()
            labels = torch.FloatTensor()
            delta_seconds = torch.LongTensor()
        else:
            q_vecs = torch.cat(q_vecs, dim=0)
            labels = torch.cat(labels, dim=0)
            delta_seconds = torch.cat(delta_seconds, dim=0)

        adataset = AnswerDataset(self.dataset.get_answer_features, self.user_list, time)
        aloader = DataLoader(adataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        a_vecs = []
        with tqdm(aloader) as pbar:
            for answer_feature in pbar:
                for feature_name in answer_feature:
                    answer_feature[feature_name] = answer_feature[feature_name].to(self.device, non_blocking=True)
                with torch.no_grad():
                    a_vec = self.model.answer_encoder(answer_feature)
                a_vecs.append(a_vec.to("cpu"))
                pbar.set_description(f"making answer_vector at {time}")
        if len(a_vecs) == 0:
            a_vecs = torch.FloatTensor()
        else:
            a_vecs = torch.cat(a_vecs, dim=0)

        return (
            q_vecs,
            a_vecs,
            delta_seconds,
            labels,
            question_ids,
        )

    def cut_df(self, begin_time, end_time):
        self.df_qas = self.df_qas[(self.df_qas["created"] >= begin_time) & (self.df_qas["created"] < end_time)]

    def ontime_users(self, begin_time, end_time):
        target_df = self.df_qas[(self.df_qas["created"] >= begin_time) & (self.df_qas["created"] < end_time)]
        users = target_df["user_id"].unique().tolist()
        return users

    @staticmethod
    def _save_vectors(vec_save_folder, time_str, vectors):
        with gzip.open(vec_save_folder + f"/{time_str}.pkl.gzip", "wb") as f:
            pickle.dump(vectors, f)

    def evaluate(
        self,
        time_begin,
        time_end,
        batch_size,
        num_workers,
        split_time_delta=timedelta(minutes=1),
        target_time=timedelta(hours=12),
        log_span=timedelta(minutes=10),
        vec_save_folder=None,
        do_evaluate=True,
    ):
        self.cut_df(time_begin - log_span, time_end)
        tmp_time = time_begin + target_time
        self.times = []
        self.aucs = []
        self.maps = []
        with TPE(max_workers=16) as exe:
            while (tmp_time >= time_begin) and (tmp_time < time_end):
                self.user_list = self.ontime_users(tmp_time - log_span, tmp_time)
                self.dict_user = {y: x for x, y in enumerate(self.user_list)}
                qvec, uvec, delta_seconds, qa_label, question_ids = self.get_vectors(
                    tmp_time, batch_size, num_workers, target_time
                )
                if (qvec.shape[0] != 0) & (uvec.shape[0] != 0):  # わからん
                    self.times.append(tmp_time)
                    if not (vec_save_folder is None):
                        vectors = {
                            "question": qvec,
                            "user": uvec,
                            "label": qa_label,
                            "delta_seconds": delta_seconds,
                            "question_ids": question_ids,
                            "user_ids": self.user_list,
                        }
                        """
                        qn, un, dim = 100, 200, 128
                        vectors["question"] = torch.rand((qn, dim))
                        vectors["user"] = torch.rand((un, dim))
                        vectors["label"] = torch.randint(0, 2, (qn, un))
                        """
                        time_str = tmp_time.strftime("%Y-%m-%d_%H_%M_%S")
                        exe.submit(self._save_vectors, vec_save_folder, time_str, vectors)
                    if do_evaluate:
                        auc_score = self.AUC(qvec, uvec, qa_label)
                        self.aucs.append(auc_score)
                        map_score = self.MAP(qvec, uvec, qa_label)
                        self.maps.append(map_score)

                tmp_time += split_time_delta
        return self.times, self.aucs, self.maps


def precision_at_k(label, score, k):
    topk_id = torch.argsort(score, -1, descending=True)
    precision = torch.gather(label, -1, topk_id)
    precision = precision[:, :k]
    out = precision.sum() / (precision.shape[0] * k)
    return out


def recall_at_k(label, score, k):
    sorted_id = torch.argsort(score, -1, descending=True)
    top_k = torch.gather(label, -1, sorted_id)[:, :k]
    recall = top_k.sum(dim=-1) / label.sum(dim=-1)
    return recall.mean()


def evaluate_from_vec(vec_folder, prefix, at_ks=[5, 50], use_gzip=True):
    if use_gzip:
        files = glob.glob(vec_folder + f"/{prefix}*.pkl.gzip")
    else:
        files = glob.glob(vec_folder + f"/{prefix}*.pkl")
    evals = []
    for file in files:
        if use_gzip:
            with gzip.open(file, "rb") as f:
                vecs_dict = pickle.load(f)
        else:
            with open(file, "rb") as f:
                vecs_dict = pickle.load(f)

        q_vec = vecs_dict["question"]
        u_vec = vecs_dict["user"]
        score = torch.matmul(u_vec, torch.transpose(q_vec, 0, 1))
        labels = torch.transpose(vecs_dict["label"], 0, 1)
        is_answered = labels.sum(axis=1) > 0
        score_target = score[is_answered, :]
        label_target = labels[is_answered, :]
        if is_answered.sum() == 0:
            continue
        map = average_precision_score(label_target, score_target, average="samples")
        if np.isnan(map):
            print(label_target, score_target)

        ndcgs = []
        pks = []
        for at_k in at_ks:
            if label_target.shape[1] < 2:
                ndcgs.append(np.nan)
            else:
                ndcgs.append(ndcg_score(label_target, score_target, k=at_k))
            pks.append(recall_at_k(label_target, score_target, at_k))
        time = file.split("/")[-1].split(".")[0]
        evals.append([time, float(map), *[float(ndcg) for ndcg in ndcgs], *[float(pk) for pk in pks]])
    out = pd.DataFrame(
        evals, columns=["time", "map", *[f"ndcg_at_{k}" for k in at_ks], *[f"recall_at_{k}" for k in at_ks]]
    )
    return out


"""
取得したいデータ
各時間(2021/7/23 0:00~1:00 )で投稿された質問のベクトル一覧{time:torch.tensor(期間内の質問数, 埋め込みベクトル次元)}
各時間(2021/7/23 0:00~1:00 など)における全ユーザーのベクトル一覧 {time:torch.tensor(ユーザー数, 埋込ベクトル次元)}
各質問において回答者が何番目にあるか {time: [[1,2,3],[3,4],[],]}
return torch.zeros(4)
"""


"""
実装するべき評価指標
(多ラベル分類とみなした混合行列の表示)
ユーザーごとにMean Average Precision
"""
