import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils_SSL import Dataset_for_MLAAD
from model import Model
import argparse
import os
from tqdm import tqdm

def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix from keys if present."""
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove the prefix 'module.'
        new_state_dict[name] = v
    return new_state_dict

def evaluate(model, dataloader, device):
    model.eval()
    scores = []
    utt_list = []

    with torch.no_grad():
        for batch_x, utt_id in tqdm(dataloader):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            # Considerando spoof = 1
            score = probs[:, 1].detach().cpu().numpy()
            scores.extend(score.tolist())
            utt_list.extend(utt_id)

    return utt_list, scores

def save_results(utt_list, scores, output_file):
    with open(output_file, 'w') as f:
        for utt_id, score in zip(utt_list, scores):
            f.write(f"{utt_id} {score:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLAAD Evaluation Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--protocol_file', type=str, required=True, help='Path to MLAAD protocol file')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of MLAAD dataset')
    parser.add_argument('--output_file', type=str, required=True, help='Output file for scores')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print("Loading model...")
    model = Model()  # Certifique-se de que esta chamada bate com sua definição

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint = remove_module_prefix(checkpoint)

    model.load_state_dict(checkpoint)
    model = model.to(args.device)

    print("Preparing dataset...")
    test_dataset = Dataset_for_MLAAD(protocol_file=args.protocol_file, data_root=args.data_root)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Running evaluation...")
    utt_list, scores = evaluate(model, test_loader, args.device)

    print(f"Saving scores to {args.output_file}...")
    save_results(utt_list, scores, args.output_file)

    print("Evaluation completed.")
