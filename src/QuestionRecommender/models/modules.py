import torch
from torch import Tensor, nn
from torch.nn.modules import dropout


class LastLayer(nn.Module):
    def __init__(self, dim_hidden, dim_out, layer_length):
        super(LastLayer, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for _ in range(layer_length)])
        self.last = nn.Linear(dim_hidden, dim_out)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = torch.relu(layer(hidden))
        out = self.last(hidden)
        return out


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class RelationHist(nn.Module):
    def __init__(
        self,
        emb_layer: nn.Module,
        dim_user: int,
        user_trans_head: int,
        dff: int,
        rate: float,
        n_layer: int,
    ) -> None:
        super(RelationHist, self).__init__()

        self.emb = emb_layer
        trans_layer = nn.TransformerEncoderLayer(
            d_model=dim_user, nhead=user_trans_head, dim_feedforward=dff, dropout=rate
        )
        self.trans = nn.TransformerEncoder(trans_layer, num_layers=n_layer)

    def forward(
        self,
        to_users: Tensor,
    ) -> Tensor:
        user_vec = self.emb(to_users)
        user_vec = torch.transpose(user_vec, 0, 1)
        out = torch.mean(self.trans(user_vec), dim=0)
        return out


class UserhistEncoder(nn.Module):
    def __init__(
        self,
        dim_user: int,
        user_trans_head: int,
        n_layer: int,
        dim_gen: int,
        dim_out: int,
        num_numerical: int,
    ) -> None:
        super(UserhistEncoder, self).__init__()

        trans_layer = nn.TransformerEncoderLayer(d_model=dim_user, nhead=user_trans_head, dim_feedforward=dim_user * 4)
        self.trans = nn.TransformerEncoder(trans_layer, num_layers=n_layer)
        self.linear = nn.Linear(num_numerical + dim_user + dim_gen, dim_out)

    def forward(
        self,
        num_user: Tensor,
        user_id: Tensor,
        user_gen: Tensor,
        to_users: Tensor,
    ) -> Tensor:
        user_id = torch.unsqueeze(user_id, 1)
        users = torch.cat((user_id, to_users), dim=1)

        out = torch.mean(self.trans(users), dim=1)
        out = self.linear(torch.cat((out, num_user, user_gen), dim=1))
        return out


class HWEncoder(nn.Module):
    def __init__(
        self,
        dim_hour: int,
        dim_weekday: int,
    ) -> None:
        super(HWEncoder, self).__init__()

        self.emb_hour = nn.Embedding(24, dim_hour)
        self.emb_weekday = nn.Embedding(7, dim_weekday)

    def forward(
        self,
        hour_weekday,
    ) -> Tensor:
        hour = hour_weekday[:, 0]
        weekday = hour_weekday[:, 1]

        hour = self.emb_hour(hour)
        weekday = self.emb_weekday(weekday)

        return torch.cat([hour, weekday], dim=1)


class CHILDEncoder(nn.Module):
    def __init__(
        self,
        dim_sex: int,
        dim_b_month: int,
    ) -> None:
        super(CHILDEncoder, self).__init__()

        self.emb_sex = nn.Embedding(3, dim_sex)  # 男、女、子供情報なしor性別不明
        self.emb_b_month = nn.Embedding(13, dim_b_month)  # 12月＋子供情報なし

    def forward(
        self,
        child_info,
    ) -> Tensor:
        sex = child_info[:, 0]
        b_month = child_info[:, 1]

        sex = self.emb_sex(sex)
        b_month = self.emb_b_month(b_month)

        return torch.cat([sex, b_month], dim=1)
