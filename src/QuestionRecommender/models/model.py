from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from .modules import CHILDEncoder, HWEncoder, Identity, LastLayer, RelationHist, UserhistEncoder

FeatureName = str
ParamName = str
"""
各特徴量の"dim_out"に出力の次元を入れるようにそろえてください
question_inputs{}
dataset = Dataset(use_features = ["user_id","text_length","category"])
dataset.__getitem__(1)
question_inputs = {
    "user_id":torch.tensor(10),
    "text_length":torch.tensor(24.),
    "category":torch.tensor(3, 12, 42)
}
"""


class VariableModel(nn.Module):
    def __init__(
        self,
        question_features: Dict[FeatureName, Dict[ParamName, int]],
        answer_features: Dict[FeatureName, Dict[ParamName, int]],
        last_features: Dict[str, int],
    ) -> None:
        super(VariableModel, self).__init__()
        global_layer = {}  # 質問と回答で重みを共有している特徴量

        # ユーザーIDの特徴量
        if ("user_id" in question_features) or ("user_id" in answer_features):
            assert question_features["user_id"] == answer_features["user_id"]
            emb_user = nn.Embedding(question_features["user_id"]["num_user"], question_features["user_id"]["dim_out"])
            global_layer["user_id"] = emb_user

        # ユーザーの世代IDの特徴量
        if ("generation_id" in question_features) or ("generation_id" in answer_features):
            assert question_features["generation_id"] == answer_features["generation_id"]
            emb_generation = nn.Embedding(
                question_features["generation_id"]["num_id"], question_features["generation_id"]["dim_out"]
            )
            global_layer["generation_id"] = emb_generation

        # ユーザーの居住地IDの特徴量
        if ("prefecture_id" in question_features) or ("prefecture_id" in answer_features):
            assert question_features["prefecture_id"] == answer_features["prefecture_id"]
            emb_prefecture = nn.Embedding(
                question_features["prefecture_id"]["num_id"], question_features["prefecture_id"]["dim_out"]
            )
            global_layer["prefecture_id"] = emb_prefecture

        # ユーザーの登録日の特徴量
        if ("register_days" in question_features) or ("register_days" in answer_features):
            assert question_features["register_days"] == answer_features["register_days"]
            global_layer["register_days"] = Identity()

        # ユーザーの末っ子の性別、誕生月の特徴量
        if ("child_info" in question_features) or ("child_info" in answer_features):
            assert question_features["child_info"] == answer_features["child_info"]
            assert (
                question_features["child_info"]["dim_sex"] + question_features["child_info"]["dim_b_month"]
                == question_features["child_info"]["dim_out"]
            )
            global_layer["child_info"] = CHILDEncoder(
                question_features["child_info"]["dim_sex"], question_features["child_info"]["dim_b_month"]
            )

        # ユーザーの末っ子の誕生年の特徴量
        if ("b_year" in question_features) or ("b_year" in answer_features):
            assert question_features["b_year"] == answer_features["b_year"]
            global_layer["b_year"] = Identity()

        self.global_layer = nn.ModuleDict(global_layer)

        # 質問、回答側のエンコーダ設定
        self.question_encoder = QuestionEncoder(question_features, self.global_layer, last_features)
        self.answer_encoder = AnswerEncoder(answer_features, self.global_layer, last_features)

    def forward(
        self,
        question_input: Dict[FeatureName, torch.Tensor],
        answer_input: Dict[FeatureName, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_vec = self.question_encoder(question_input)
        a_vec = self.answer_encoder(answer_input)
        return q_vec, a_vec


class QuestionEncoder(nn.Module):
    def __init__(
        self,
        features: Dict[FeatureName, Dict[ParamName, int]],
        global_layer: nn.ModuleDict,
        last_features: Dict[str, int],
    ) -> None:
        super(QuestionEncoder, self).__init__()
        self.global_layer = global_layer
        local_layer = {}
        if "q_length" in features:
            local_layer["q_length"] = Identity()
        if "category_id" in features:
            local_layer["category_id"] = nn.Embedding(
                features["category_id"]["num_id"], features["category_id"]["dim_out"]
            )
        if "hour_and_weekday" in features:
            assert (
                features["hour_and_weekday"]["dim_hour"] + features["hour_and_weekday"]["dim_weekday"]
                == features["hour_and_weekday"]["dim_out"]
            )
            local_layer["hour_and_weekday"] = HWEncoder(
                features["hour_and_weekday"]["dim_hour"], features["hour_and_weekday"]["dim_weekday"]
            )
        if "word2vec" in features:
            local_layer["word2vec"] = Identity()
        if "word_count" in features:
            local_layer["word_count"] = Identity()
        self.local_layer = nn.ModuleDict(local_layer)
        dim_hidden = 0
        for feature in features:
            dim_hidden += features[feature]["dim_out"]
        if last_features is None:
            self.last_layer = Identity()
        elif isinstance(last_features, int):
            self.last_layer = nn.Linear(dim_hidden, last_features)
        else:
            self.last_layer = LastLayer(dim_hidden, last_features["dim_out"], last_features["layer_length"])

    def forward(
        self,
        question_input: Dict[FeatureName, torch.Tensor],
    ) -> torch.Tensor:
        hiddens = []
        for feats in self.global_layer:
            hiddens.append(torch.squeeze(self.global_layer[feats](question_input[feats]), dim=1))
        for feats in self.local_layer:
            hiddens.append(torch.squeeze(self.local_layer[feats](question_input[feats]), dim=1))
        out = self.last_layer(torch.cat(hiddens, dim=-1))
        return out


class AnswerEncoder(nn.Module):
    def __init__(
        self,
        features: Dict[FeatureName, Dict[ParamName, int]],
        global_layer: nn.ModuleDict,
        last_features: Dict[str, int],
    ) -> None:
        super(AnswerEncoder, self).__init__()
        self.global_layer = global_layer
        local_layer = {}
        if "to_user_id" in features:
            if features["to_user_id"]["shareweights"]:
                emblayer = self.global_layer["user_id"]
            else:
                emblayer = nn.Embedding(features["to_user_id"]["num_user"], features["to_user_id"]["dim_out"])
            local_layer["to_user_id"] = RelationHist(
                emblayer,
                features["to_user_id"]["dim_out"],
                features["to_user_id"]["num_head"],
                features["to_user_id"]["dff"],
                features["to_user_id"]["dropout"],
                features["to_user_id"]["n_layer"],
            )
        if "hist_features" in features:
            local_layer["hist_features"] = Identity()
        if "avg_hist_q_embs" in features:
            local_layer["avg_hist_q_embs"] = Identity()

        self.local_layer = nn.ModuleDict(local_layer)

        dim_hidden = 0
        for feature in features:
            dim_hidden += features[feature]["dim_out"]
        if last_features is None:
            self.last_layer = Identity()
        elif isinstance(last_features, int):
            self.last_layer = nn.Linear(dim_hidden, last_features)
        else:
            self.last_layer = LastLayer(dim_hidden, last_features["dim_out"], last_features["layer_length"])

    def forward(
        self,
        answer_input: Dict[FeatureName, torch.Tensor],
    ) -> torch.Tensor:
        hiddens = []
        for feats in self.global_layer:
            hiddens.append(torch.squeeze(self.global_layer[feats](answer_input[feats]), dim=1))
        for feats in self.local_layer:
            hiddens.append(torch.squeeze(self.local_layer[feats](answer_input[feats]), dim=1))
        out = self.last_layer(torch.cat(hiddens, dim=-1))
        return out


class BaselineModel(nn.Module):
    def __init__(
        self,
        num_user: int,
        dim_user: int,
    ) -> None:
        super(BaselineModel, self).__init__()

        self.emb_user = nn.Embedding(num_user, dim_user)

    def forward(
        self,
        cat_q: Tensor,  # 使わない
        num_q: Tensor,  # 使わない
        cat_qu: Tensor,
        num_qu: Tensor,  # 使わない
        cat_au: Tensor,
        num_au: Tensor,  # 使わない
    ) -> Tuple[Tensor, Tensor]:

        cat_qu = torch.squeeze(cat_qu, dim=1)
        cat_au = torch.squeeze(cat_au, dim=1)

        q_out = self.emb_user(cat_qu)
        a_out = self.emb_user(cat_au)

        return q_out, a_out


class UserCategoryModel(nn.Module):
    def __init__(
        self,
        num_user: int,
        num_category: int,
        num_gen: int,
        dim_user: int,  # dimは埋め込みベクトルの次元
        dim_category: int,
        dim_gen: int,
        dim_hour: int,
        dim_week: int,
        dim_num: int,
        dim_out: int,
    ) -> None:
        super(UserCategoryModel, self).__init__()

        self.emb_user = nn.Embedding(num_user, dim_user)
        self.emb_gen = nn.Embedding(num_gen, dim_gen)
        self.emb_cat = nn.Embedding(num_category, dim_category)
        self.emb_hour = nn.Embedding(24, dim_hour)
        self.emb_week = nn.Embedding(7, dim_week)

        self.ffquestion1 = nn.Linear(
            dim_user + dim_gen + dim_category + dim_hour + dim_week + dim_num, dim_out
        )  # 埋め込みの次元
        self.ffquestion2 = nn.Linear(dim_out, dim_out)
        self.ffanswer1 = nn.Linear(dim_user + dim_gen, dim_out)
        self.ffanswer2 = nn.Linear(dim_out, dim_out)

    def forward(
        self,
        cat_q: Tensor,
        num_q: Tensor,  # 使わない
        question_user: Tensor,
        num_qu: Tensor,  # 使わない
        answer_user: Tensor,
        num_au: Tensor,  # 使わない
    ) -> Tuple[Tensor, Tensor]:
        category = cat_q[:, 0]
        hour = cat_q[:, 1]
        week = cat_q[:, 2]

        question_user_id = question_user[:, 0]
        question_user_generation = question_user[:, 1]
        answer_user_id = answer_user[:, 0]
        answer_user_generation = answer_user[:, 1]

        quser_vec = self.emb_user(question_user_id)
        qgen_vec = self.emb_gen(question_user_generation)

        c_vec = self.emb_cat(category)
        h_vec = self.emb_hour(hour)
        w_vec = self.emb_week(week)

        auser_vec = self.emb_user(answer_user_id)
        agen_vec = self.emb_gen(answer_user_generation)

        q_feature = torch.cat((quser_vec, qgen_vec, c_vec, h_vec, w_vec, num_q), dim=1)
        a_feature = torch.cat((auser_vec, agen_vec), dim=1)

        q_out = self.ffquestion2(torch.relu(self.ffquestion1(q_feature)))
        a_out = self.ffanswer2(torch.relu(self.ffanswer1(a_feature)))

        return q_out, a_out


class UserhistModel(nn.Module):
    def __init__(
        self,
        num_user: int,
        num_category: int,
        num_gen: int,
        dim_user: int,
        dim_category: int,
        dim_gen: int,
        dim_out: int,
        user_trans_head: int,
        n_layer: int,
        num_numerical: int,
    ) -> None:
        super(UserhistModel, self).__init__()

        self.emb_user = nn.Embedding(num_user, dim_user)
        self.emb_gen = nn.Embedding(num_gen, dim_gen)
        self.emb_cat = nn.Embedding(num_category, dim_category)

        self.ffquestion1 = nn.Linear(dim_user + dim_gen + dim_category, dim_out)
        self.ffquestion2 = nn.Linear(dim_out, dim_out)
        self.ffanswer1 = nn.Linear(dim_user + dim_gen, dim_out)
        self.ffanswer2 = nn.Linear(dim_out, dim_out)

        self.ans_encoder = UserhistEncoder(dim_user, user_trans_head, n_layer, dim_gen, dim_out, num_numerical)

    def forward(
        self,
        category: Tensor,
        num_q: Tensor,  # 使わない
        question_user: Tensor,
        num_qu: Tensor,  # 使わない
        cat_au: Tensor,
        num_au: Tensor,  # 使わない
    ) -> Tuple[Tensor, Tensor]:
        category = torch.squeeze(category, 1)
        question_user_id = question_user[:, 0]
        question_user_generation = question_user[:, 1]
        answer_user_id = cat_au[:, 0]
        answer_user_generation = cat_au[:, 1]
        answer_to_user = cat_au[:, 2:]
        num_au = num_au[:, 0]

        quser_vec = self.emb_user(question_user_id)
        qgen_vec = self.emb_gen(question_user_generation)
        c_vec = self.emb_cat(category)
        auser_vec = self.emb_user(answer_user_id)
        agen_vec = self.emb_gen(answer_user_generation)
        to_user_vec = self.emb_user(answer_to_user)

        q_feature = torch.cat((quser_vec, qgen_vec, c_vec), dim=1)
        q_out = self.ffquestion2(torch.relu(self.ffquestion1(q_feature)))

        a_out = self.ans_encoder(num_au, auser_vec, agen_vec, to_user_vec)

        return q_out, a_out
