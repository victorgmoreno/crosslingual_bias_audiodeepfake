import argparse
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import Model
from core_scripts.startup_config import set_random_seed

# ===========================
# Avalia modelo treinado ASVspoof em dados M-AILABS (todos bonafide) usando Dataset/DataLoader
# ===========================

class MailabsDataset(Dataset):
    def __init__(self, dataframe, target_len=64600, sample_rate=16000):
        self.dataframe = dataframe
        self.target_len = target_len
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        file_path = row['file_path']
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        waveform = waveform[0]  # mono
        if len(waveform) > self.target_len:
            waveform = waveform[:self.target_len]
        else:
            repeats = (self.target_len // len(waveform)) + 1
            waveform = waveform.repeat(repeats)[:self.target_len]
        return waveform, idx


def main(args):
    set_random_seed(args.seed, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.csv_path)
    print(f"Total de amostras: {len(df)}")

    # Dataset e DataLoader
    dataset = MailabsDataset(df)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    # Carrega modelo
    model = Model(args, device)
    state_dict = torch.load(args.model_path, map_location=device)
    if 'module.' in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPUs com DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    # Avaliação
    scores = {}
    for batch_x, batch_idx in tqdm(dataloader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            out = model(batch_x)
            score_batch = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        for i, idx in enumerate(batch_idx):
            scores[int(idx)] = float(score_batch[i])

    # Salva resultado
    df['score'] = df.index.map(scores)
    df.to_csv(args.output_csv, index=False)
    print(f"Arquivo salvo com scores: {args.output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval M-AILABS with ASVspoof model")
    parser.add_argument('--csv_path', type=str, required=True, help='Caminho do CSV de entrada')
    parser.add_argument('--output_csv', type=str, required=True, help='Caminho para salvar CSV com scores')
    parser.add_argument('--model_path', type=str, required=True, help='Checkpoint do modelo .pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cudnn_deterministic_toggle', action='store_false', default=True,
                        help='use cudnn-deterministic? (default true)')
    parser.add_argument('--cudnn_benchmark_toggle', action='store_true', default=False,
                        help='use cudnn-benchmark? (default false)')

    args = parser.parse_args()
    main(args)
