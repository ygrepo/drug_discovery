import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
from sequence_models.structure import Attention1d


class Decoder(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention1d = Attention1d(in_dim=hidden_dim)
        self.dense_3 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.dense_1(x))
        x = torch.relu(self.dense_2(x))
        x = self.attention1d(x)
        x = torch.relu(self.dense_3(x))
        x = self.dense_4(x)
        return x


class EsmForLandscapeRegression(nn.Module):
    def __init__(self, esm_path, decoder_ckpt, device):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_path)
        self.esm_tokenizer = EsmTokenizer.from_pretrained(esm_path)
        self.decoder = Decoder()
        ckpt = torch.load(decoder_ckpt)
        self.decoder.load_state_dict(ckpt)
        self.device = device

    def forward(self, protein):
        protein = self.esm_tokenizer(
            list(protein),
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt",
        ).to(self.device)
        h = self.esm(**protein, return_dict=True).last_hidden_state
        return self.decoder(h).squeeze()

    def predict_fitness(self, protein, *kwargs):
        return self.forward(protein)


# if __name__ == "__main__":
#     import os
#     import sys
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     from tqdm import tqdm
#     from dataset.fitness_dataset import FitnessDataset
#     from torch.utils.data import DataLoader
#     from sklearn.metrics import r2_score
#     from scipy.stats import spearmanr
#     ckpt_path = "./ckpts/landscape_ckpts/landscape_params/esm1b_landscape/"
#     data_path = "./data/fitness/"
#     device = torch.device("cuda", 0)
#     for dataset_name in ["AAV", "AMIE", "avGFP", "E4B", "LGK", "Pab1", "TEM", "UBE2I"]:
#         print(dataset_name)
#         dataset = FitnessDataset(data_path + dataset_name, "valid")
#         dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
#         model = EsmForLandscapeRegression("./ckpts/protein_ckpts/esm1b", ckpt_path + dataset_name + "/decoder.pt", device).to(device)
#         preds, gts = [], []
#         model.eval()
#         with torch.no_grad():
#             for i, (seq, score) in enumerate(tqdm(dataloader)):
#                 preds += model(seq).tolist()
#                 gts += score.tolist()
#                 if i == 0:
#                     print("Preds:", model(seq).tolist())
#                     print("gts: ", score.tolist())
#         print("R2:", r2_score(gts, preds))
#         print("Spearman: ", spearmanr(gts, preds))
