import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


class SimpleTrainer:
    def __init__(self, model: Module, loader: DataLoader, criterion, optim, device="cuda"):
        """単純なトレーナー

        Args:
            model: 学習したいモデル
            loader: データローダー
            criterion: 損失関数
            optim: オプティマイザー
        """
        self.loader = loader
        self.criterion = criterion
        self.criterion.to(device)
        self.model = model
        self.model.to(device)
        self.optim = optim
        self.device = device

    def train(self, epoch):
        self.model.train()
        loss_mean = 0
        with tqdm(self.loader) as pbar:
            for i, ((question_feature, answer_feature), label) in enumerate(pbar):
                # loaderの出力をデバイスに乗せる
                for feature_name in question_feature:
                    question_feature[feature_name] = question_feature[feature_name].to(self.device, non_blocking=True)
                for feature_name in answer_feature:
                    answer_feature[feature_name] = answer_feature[feature_name].to(self.device, non_blocking=True)
                # question_feature = question_feature.to(self.device, non_blocking=True)
                # answer_feature = answer_feature.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                # modelを計算
                q_vec, a_vec = self.model(question_feature, answer_feature)

                # 内積を取って損失を見る
                out = torch.matmul(q_vec, torch.transpose(a_vec, 0, 1))
                label_matrix = torch.diag(label[:, 0]).type_as(out).to(self.device, non_blocking=True)
                loss = self.criterion(out, label_matrix)

                # 学習
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                # tqdmの表示
                loss_mean = (loss_mean * i + loss.item()) / (i + 1)
                pbar.set_description(f"epoch {epoch+1} : loss={loss_mean}")
        return loss_mean

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model = self.model.load_state_dict(torch.load(path))

    def evaluate_model(self, loader=None):
        if loader is None:
            loader = self.loader

        self.model.eval()

        with torch.no_grad():
            with tqdm(loader) as pbar:
                for i, ((question_feature, answer_feature), label) in enumerate(pbar):
                    # loaderの出力をデバイスに乗せる
                    for feature_name in question_feature:
                        question_feature[feature_name] = question_feature[feature_name].to(
                            self.device, non_blocking=True
                        )
                    for feature_name in answer_feature:
                        answer_feature[feature_name] = answer_feature[feature_name].to(self.device, non_blocking=True)
                    # question_feature = question_feature.to(self.device, non_blocking=True)
                    # answer_feature = answer_feature.to(self.device, non_blocking=True)
                    label = label.to(self.device, non_blocking=True)

                    # modelを計算
                    q_vec, a_vec = self.model(question_feature, answer_feature)

                    # 内積を取って損失を見る
                    out = torch.matmul(q_vec, torch.transpose(a_vec, 0, 1))
                    label_matrix = torch.diag(label[:, 0]).type_as(out).to(self.device, non_blocking=True)
                    loss = self.criterion(out, label_matrix)

                    loss_mean = (loss_mean * i + loss.item()) / (i + 1)

                    # tqdmの表示
                    pbar.set_description(f"valid_loss={loss_mean}")
