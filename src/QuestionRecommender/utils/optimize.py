import glob
import os
import pickle
from collections import OrderedDict, deque
from datetime import datetime as dt
from typing import *

import numba as nb
import numpy as np
import pandas as pd
from loguru import logger
from numba import njit, typed, types
from numba.experimental import jitclass
from tqdm import tqdm


def loadPickle(fileName):
    with open(fileName, mode="rb") as f:
        return pickle.load(f)


def f(n: float) -> float:
    return np.log1p(n)


def list_pickles(save_dir: str) -> Dict["datetime", "Output"]:
    out = dict()
    for fname in glob.glob(os.path.join(save_dir, "*.pkl")):
        now = dt.strptime(fname.split("/")[-1].replace(".pkl", ""), "%Y-%m-%d_%H_%M")
        out[now] = loadPickle(fname)
    _out = OrderedDict()
    for key in sorted(out.keys()):
        _out[key] = out[key]
    for key in _out.keys():
        yield key, _out[key]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def off(qvec, uvec):
    out = np.log(sigmoid(qvec @ uvec.T)).mean()
    return -2.34065436425 - out


def load_output(output):
    output["question_ids"] = list(output["question_ids"])
    q_vec = output["question"].numpy()
    u_vec = output["user"].numpy()
    label = output["label"].numpy()
    q_ids = np.array(output["question_ids"])
    u_ids = np.array(output["user_ids"])
    return q_vec, u_vec, label, q_ids, u_ids


kv_ty = (nb.typeof((1, 1)), nb.float64)


@nb.njit(parallel=True, fastmath=True)
def cali_p(qvec, uvec, q_ids, u_ids, offset):
    logits = qvec @ uvec.T

    # logシグモイド関数にoffsetで補正をかける
    logp = np.log(1.0 / (1.0 + np.exp(-logits))) + offset
    P = np.exp(logp)  # なぜexpをつけるのか忘れた．
    cali_p = typed.Dict.empty(*kv_ty)
    for i in range(len(q_ids)):
        for j in range(len(u_ids)):
            q = q_ids[i]
            u = u_ids[j]
            cali_p[(q, u)] = P[i, j]
    return cali_p


kv_ty2 = (nb.typeof(1), nb.float64)


@nb.njit(fastmath=True)
def PI(P, q_ids, beta=100):
    pi = typed.Dict.empty(*kv_ty)
    bottom = typed.Dict.empty(*kv_ty2)
    for (q, u), p in P.items():
        if u not in bottom:
            s = 0.0
            for i in q_ids:
                s += P[i, u] ** beta
            bottom[u] = s
        else:
            s = bottom[u]
        top = P[q, u] ** beta
        pi[(q, u)] = top / s
    return pi


@nb.njit(fastmath=True)
def _all_item3(P, pi, q_ids, u_ids):
    auser_item3 = typed.Dict.empty(*kv_ty2)
    for q in q_ids:
        out = 0.0
        for user in u_ids:
            out += P[q, user] * pi[q, user]
        auser_item3[q] = out
    return auser_item3


def get_ans_user(dfa, q_ids, time):
    logger.info("getting ansered users for {}".format(time))
    df = dfa[dfa["created"] < time]
    out = df.groupby("question_id").user_id.unique().to_dict()
    for q in q_ids:
        if q in out:
            out[q] = set(out[q])
        else:
            out[q] = set()
    logger.info("done")
    return out


class Opt:
    count: int = 0
    U_i: Dict["user_id", "deque[question_id]"] = dict()

    def __init__(
        self,
        q_vec,
        u_vec,
        label,
        q_ids,
        u_ids,
        U_a,
        k=5,
        max_len=8000,
        beta=100,
        lamda=1,
        term1=False,
        term2=True,
        term3=False,
    ):
        Opt.count += 1
        self.q_vec = q_vec
        self.u_vec = u_vec
        self.label = label
        self.q_ids = q_ids
        self.u_ids = u_ids
        self.q_len = q_vec.shape[0]
        self.u_len = u_vec.shape[0]
        self.U_a = U_a
        self.max_len = max_len
        self.beta = beta
        self.lamda = lamda
        self.term1 = term1
        self.term2 = term2
        self.term3 = term3
        self.k = k

    def calibrate(self):
        logger.info("calibrating...")
        offset = off(self.q_vec, self.u_vec)
        self.P = cali_p(self.q_vec, self.u_vec, self.q_ids, self.u_ids, offset)
        logger.info("done")

    def calculate_pi(self):
        logger.info("calculating pi...")
        self.pi = PI(self.P, self.q_ids, beta=self.beta)
        logger.info("done")

    def all_item3(self):
        logger.info("calculating all_item3...")
        self.auser_item3 = _all_item3(self.P, self.pi, self.q_ids, self.u_ids)
        logger.info("done")

    def typedDict2dict(self):
        logger.info("typedDict2dict...")
        self.P = dict(self.P)
        self.pi = dict(self.pi)
        self.auser_item3 = dict(self.auser_item3)
        logger.info("done")

    def item1(self, q_id, u_id):
        out = len(self.U_a[q_id] - set([u_id]))
        return out

    def item2(self, q_id, u_id):
        if q_id not in Opt.U_i:
            Opt.U_i[q_id] = deque([], self.max_len)
        u = {u for u in Opt.U_i[q_id] if u in self.u_ids}
        u = u - self.U_a[q_id] - set([u_id])
        out = np.array([self.P[q_id, i] for i in u]).sum()
        return out

    def item3(self, q_id, u_id):
        if q_id not in Opt.U_i:
            Opt.U_i[q_id] = deque([], self.max_len)
        out = 0
        u_i = {u for u in Opt.U_i[q_id] if u in self.u_ids}
        for other_user in u_i:  # 過去レコメンドした人のうち現在いるユーザーのみ
            out += self.P[q_id, other_user] * self.pi[q_id, other_user]
        out += self.P[q_id, u_id] * self.pi[q_id, u_id]
        out = self.auser_item3[q_id] - out
        return out

    def delta_phi(self, q_id, u_id):
        item = 0
        if self.term1:
            item += self.item1(q_id, u_id)
        if self.term2:
            item += self.item2(q_id, u_id)
        if self.term3:
            item += self.item3(q_id, u_id)
        out = f(item + self.P[q_id, u_id]) - f(item)  # 第4項が1の場合から0の場合の差分
        return out

    def recommend(self):
        logger.info("recommending...")
        self.topk = dict()
        for u in tqdm(self.u_ids):
            deltas = np.zeros(len(self.q_ids))
            for i, q in enumerate(self.q_ids):
                deltas[i] = self.delta_phi(q, u)
            self.topk[u] = self.q_ids[np.argsort(deltas)[-self.k :]].tolist()
        logger.info("done")

    def update(self):
        logger.info("updating...")
        for u, r_ids in tqdm(self.topk.items()):
            for q in r_ids:
                if q not in Opt.U_i:
                    Opt.U_i[q] = deque([], self.max_len)
                Opt.U_i[q].append(u)
        logger.info("done")

    def evaluate(self):
        logger.info("evaluating...")
        sum_f = 0
        hit = 0
        for i, q in tqdm(enumerate(self.q_ids), total=len(self.q_ids)):
            count = 0
            for j, u in enumerate(self.u_ids):
                if self.label[i, j] > 0 and q in self.topk[u]:
                    count += 1
            sum_f += f(count)
            hit += count
        result = {"sum_f": sum_f, "hit": hit}
        logger.info("done")
        return result

    def __call__(self):
        self.calibrate()
        self.calculate_pi()
        self.all_item3()
        # self.typedDict2dict()
        self.recommend()
        self.update()
        return self.evaluate()


if __name__ == "__main__":
    save_dir = "evaluate"
    dfa = pd.read_table("answers_2021.tsv", usecols=["question_id", "user_id", "created"])
    dfa["created"] = pd.to_datetime(dfa["created"], format="%Y-%m-%d %H:%M:%S")
    for now, out in list_pickles(save_dir):
        U_a = get_ans_user(dfa, out["question_ids"], now)
        opt = Opt(*load_output(out), U_a)
        result = opt()
        logger.info(now)
        logger.info(result)
