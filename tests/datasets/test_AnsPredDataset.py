import unittest
from datetime import datetime, timedelta

import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from QuestionRecommender.datasets.dataset import AnsPredDataset
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
from QuestionRecommender.models.model import VariableModel
from QuestionRecommender.utils.evaluate import Evaluater, OnTimeEvaluater, evaluate_from_vec
from QuestionRecommender.utils.trainer import SimpleTrainer


class TestAnsPredDataset(unittest.TestCase):
    """test AnsPredDataset"""

    def test_dataset(self):
        """test build function"""
        # データの読み込み
        df_question_law = pd.read_pickle("sample_data/question_with_vec.pkl")
        df_answer_law = pd.read_csv("sample_data/answer.csv")
        df_user_law = pd.read_csv("sample_data/users.csv")
        df_children_law = pd.read_csv("sample_data/children.csv")
        df_qas = pd.read_csv("sample_data/user_q&a&s.csv")
        path = "src/QuestionRecommender/datasets/sample.model"

        # データの前処理
        df_merged = merge_question_answer(df_question_law, df_answer_law)
        question_categorical, question_numerical, oe_q = get_question_categoty(df_question_law)
        df_uc = merge_user_children(df_user_law, df_children_law)
        user_categorical, user_numerical, oe_u = get_user_generation(df_uc)

        df_qas_nc = get_qas_nocontent(df_qas)
        df_ans_nc = get_ans_nocontent(df_answer_law)
        ans_add_to_user = get_to_userid(df_ans_nc, df_question_law)
        qas_nc_add_to_user = merge_qas_ans_add_to_user(df_qas_nc, ans_add_to_user)
        qas_nc_add_to_user_dict = to_dict_qas_ans_add_to_user(qas_nc_add_to_user)

        use_features = [
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

        # データセットの作成
        dataset = AnsPredDataset(
            df_merged,
            question_categorical,
            question_numerical,
            user_categorical,
            user_numerical,
            qas_nc_add_to_user_dict,
            use_features,
            sentence_embedding_dim=100,
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

        # データセットの性質テスト
        self.assertEqual(len(df_merged), len(dataset))
        for i, tmp_data in enumerate(loader):
            self.assertEqual(2, len(tmp_data))
            feats, labels = tmp_data
            question_features, answer_features = feats
            self.assertTrue(question_features["category_id"].dtype is torch.long)
            self.assertTrue(question_features["hour_and_weekday"].dtype is torch.long)
            self.assertTrue(question_features["word2vec"].dtype is torch.float)
            self.assertTrue(question_features["word_count"].dtype is torch.float)
            self.assertTrue(question_features["user_id"].dtype is torch.long)
            self.assertTrue(question_features["generation_id"].dtype is torch.long)
            self.assertTrue(question_features["prefecture_id"].dtype is torch.long)
            self.assertTrue(question_features["register_days"].dtype is torch.float)
            self.assertTrue(question_features["child_info"].dtype is torch.long)
            self.assertTrue(question_features["b_year"].dtype is torch.float)
            self.assertTrue(answer_features["user_id"].dtype is torch.long)
            self.assertTrue(answer_features["generation_id"].dtype is torch.long)
            self.assertTrue(answer_features["to_user_id"].dtype is torch.long)
            self.assertTrue(answer_features["hist_features"].dtype is torch.float)
            self.assertTrue(labels.dtype is torch.bool)

        num_u = oe_u.get_params()["mapping"][0]["mapping"].max() + 1
        num_c = oe_q.get_params()["mapping"][0]["mapping"].max() + 1
        num_g = 8
        num_p = 48  # 47都道府県+情報なし

        question_features = {
            "category_id": {"num_id": num_c, "dim_out": 2},
            "hour_and_weekday": {
                "dim_hour": 2,
                "dim_weekday": 2,
                "dim_out": 4,
            },
            "word2vec": {"dim_out": 100},
            "word_count": {"dim_out": 1},
            "user_id": {"num_user": num_u, "dim_out": 2},
            "generation_id": {"num_id": num_g, "dim_out": 2},
            "prefecture_id": {"num_id": num_p, "dim_out": 2},
            "register_days": {"dim_out": 1},
            "child_info": {
                "dim_sex": 2,
                "dim_b_month": 2,
                "dim_out": 4,
            },
            "b_year": {"dim_out": 1},
        }
        answer_features = {
            "user_id": {"num_user": num_u, "dim_out": 2},
            "generation_id": {"num_id": num_g, "dim_out": 2},
            "prefecture_id": {"num_id": num_p, "dim_out": 2},
            "register_days": {"dim_out": 1},
            "child_info": {
                "dim_sex": 2,
                "dim_b_month": 2,
                "dim_out": 4,
            },
            "b_year": {"dim_out": 1},
            "to_user_id": {
                "shareweights": True,
                "num_user": num_u,  # 実質使っていない
                "dim_out": 2,
                "num_head": 1,
                "dff": 1,
                "dropout": 0.1,
                "n_layer": 2,
            },
            "hist_features": {"dim_out": 24},
            "avg_hist_q_embs": {"dim_out": 100},
        }
        last_features = {"dim_out": 2, "layer_length": 1}
        model = VariableModel(question_features, answer_features, last_features)
        criterion = BCEWithLogitsLoss(pos_weight=torch.ones(1) * 1)
        optim = torch.optim.Adam(model.parameters())

        trainer = SimpleTrainer(model, loader, criterion, optim, device="cpu")
        for i in range(2):
            trainer.train(i)

        evaluater = Evaluater(dataset, trainer.model, df_merged, df_user_law, device="cpu")
        b = datetime(2019, 1, 1, 0, 0)
        e = datetime(2019, 1, 1, 0, 5)
        times, aucs, maps = evaluater.evaluate(b, e, batch_size=2, num_workers=0)

        evaluater = OnTimeEvaluater(dataset, trainer.model, df_merged, df_qas, device="cpu")
        times, aucs, maps = evaluater.evaluate(
            b,
            e,
            batch_size=2,
            num_workers=0,
            split_time_delta=timedelta(seconds=10),
            target_time=timedelta(minutes=1),
            vec_save_folder=None,  # "experiments"
            do_evaluate=True,
        )
        # df = evaluate_from_vec("experiments", "")
        # df.to_csv("experiments/eval.csv")


if __name__ == "__main__":
    unittest.main()
