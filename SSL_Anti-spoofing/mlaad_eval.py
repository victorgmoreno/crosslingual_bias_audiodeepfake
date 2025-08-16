import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from model import Model

# --- SETTINGS ---
mlaad_root = '/home/victor.moreno/dl-29_backup/spoof/dataset/mlaad'
model_path = '/home/victor.moreno/dl-29_backup/spoof/SSL_Anti-spoofing/models/model_weighted_CCE_100_14_1e-06_asvspoof5_training/epoch_9.pth'
output_csv = 'mlaad_evaluation_results.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Define minimal args needed by your model ---
class Args:
    def __init__(self):
        self.algo = 0  # Set any other args if your model uses them internally

args = Args()

# --- Initialize and load model ---
print("Loading model...")
model = Model(args=args, device=device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded and ready.")

# --- Helper: Process and resample audio to 16kHz ---
def process_audio(audio_path, target_sr=16000, max_len=64600):
    wav, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    # Pad or crop to fixed length (64600 samples ≈ 4 sec)
    if wav.shape[1] < max_len:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]))
    else:
        wav = wav[:, :max_len]
    return wav

# --- Scan MLAAD dataset and collect files ---
results = []
print(f"Scanning MLAAD dataset in: {mlaad_root}")

for lang_folder in tqdm(os.listdir(mlaad_root), desc='Languages'):
    lang_path = os.path.join(mlaad_root, lang_folder)
    if not os.path.isdir(lang_path):
        continue

    for model_folder in os.listdir(lang_path):
        model_path_folder = os.path.join(lang_path, model_folder)
        meta_file = os.path.join(model_path_folder, 'meta.csv')

        if not os.path.isfile(meta_file):
            print(f'⚠️ Warning: No meta.csv found in {model_path_folder}')
            continue

        meta_df = pd.read_csv(meta_file, sep='|')

        for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f'{lang_folder}/{model_folder}'):
            audio_file = os.path.join(model_path_folder, row['path'])
            if not os.path.isfile(audio_file):
                print(f'⚠️ Missing file: {audio_file}')
                continue

            # Process audio (resample + pad/crop)
            wav = process_audio(audio_file)
            wav = wav.to(device)

            # Run model
            with torch.no_grad():
                out = model(wav)
                # Binary spoof score: take probability for class 1 (spoof)
                score = torch.softmax(out, dim=1)[:, 1].cpu().item()

            # Save result
            results.append({
                'file_path': audio_file,
                'language': row.get('language', ''),
                'model_name': row.get('model_name', ''),
                'architecture': row.get('architecture', ''),
                'score': score,
                'duration': row.get('duration', ''),
                'is_original_language': row.get('is_original_language', ''),
                'training_data': row.get('training_data', ''),
                'original_file': row.get('original_file', ''),
                'transcript': row.get('transcript', '')
            })

# --- Save results to CSV ---
output_df = pd.DataFrame(results)
output_df.to_csv(output_csv, index=False)
print(f"\n Evaluation completed. Results saved to: {output_csv}")
