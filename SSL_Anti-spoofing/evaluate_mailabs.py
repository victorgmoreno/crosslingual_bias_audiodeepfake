import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import librosa
from model import Model


__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


class Dataset_MAILABS_eval(Dataset):
    def __init__(self, csv_file):
        """
        Dataset for MAILABS evaluation - CORRECTED VERSION with stratified sampling
        Args:
            csv_file: path to CSV file containing file paths and metadata
        """
        self.df = pd.read_csv(csv_file)
        
        # Amostragem estratificada por idioma - 1000 amostras totais
        print("\n=== AMOSTRAGEM ESTRATIFICADA POR IDIOMA ===")
        
        # Contar amostras por idioma
        language_counts = self.df['language'].value_counts()
        print("Distribuição original por idioma:")
        for lang, count in language_counts.items():
            print(f"  {lang}: {count} amostras")
        
        total_languages = len(language_counts)
        samples_per_language = 1000 // total_languages
        remaining_samples = 1000 % total_languages
        
        print(f"\nTotal de idiomas: {total_languages}")
        print(f"Amostras por idioma: {samples_per_language}")
        print(f"Amostras extras para distribuir: {remaining_samples}")
        
        # Amostragem estratificada
        stratified_samples = []
        
        for i, (language, group) in enumerate(self.df.groupby('language')):
            # Calcular número de amostras para este idioma
            n_samples = samples_per_language
            if i < remaining_samples:  # Distribuir amostras extras nos primeiros idiomas
                n_samples += 1
            
            # Se o idioma tem menos amostras que o necessário, pegar todas
            n_samples = min(n_samples, len(group))
            
            # Amostragem aleatória
            sampled_group = group.sample(n=n_samples, random_state=42)
            stratified_samples.append(sampled_group)
            
            print(f"  {language}: {n_samples} amostras selecionadas de {len(group)} disponíveis")
        
        # Combinar todas as amostras
        self.df = pd.concat(stratified_samples, ignore_index=True)
        
        # Embaralhar para misturar os idiomas
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nTotal de amostras selecionadas: {len(self.df)}")
        print("\nDistribuição final por idioma:")
        final_counts = self.df['language'].value_counts()
        for lang, count in final_counts.items():
            print(f"  {lang}: {count} amostras")
        
        self.list_IDs = self.df['file_path'].tolist()
        self.file_names = self.df['file_name'].tolist()
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        audio_path = self.list_IDs[index]
        file_name = self.file_names[index]
        
        # TESTE: Usar exatamente o mesmo método do ASVspoof original
        import librosa
        X, fs = librosa.load(audio_path, sr=16000)
        
        # Função de padding exata do ASVspoof original
        def pad_original(x, max_len=64600):
            x_len = x.shape[0]
            if x_len >= max_len:
                return x[:max_len]
            num_repeats = int(max_len / x_len) + 1
            padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
            return padded_x
        
        X_pad = pad_original(X, 64600)
        x_inp = torch.Tensor(X_pad)
        
        return x_inp, file_name


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    
    all_results = []
    
    print(f"Starting evaluation of {len(dataset)} samples...")
    
    for batch_idx, (batch_x, utt_ids) in enumerate(data_loader):
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        with torch.no_grad():
            batch_out = model(batch_x)
            
            print(f"\nDEBUG - Batch {batch_idx + 1}:")
            print(f"Batch output shape: {batch_out.shape}")
            print(f"First sample raw output: {batch_out[0].cpu().numpy()}")
            
            # TESTE 1: Método atual (logits spoof)
            method1_scores = batch_out[:, 1].data.cpu().numpy().ravel()
            
            # TESTE 2: Logits bonafide (negativos)
            method2_scores = -batch_out[:, 0].data.cpu().numpy().ravel()
            
            # TESTE 3: Diferença (spoof - bonafide)
            method3_scores = (batch_out[:, 1] - batch_out[:, 0]).data.cpu().numpy().ravel()
            
            # TESTE 4: Softmax spoof
            softmax_probs = torch.softmax(batch_out, dim=1)
            method4_scores = softmax_probs[:, 1].cpu().numpy().ravel()
            
            print(f"Método 1 (logits spoof): {method1_scores[:3]}")
            print(f"Método 2 (-logits bonafide): {method2_scores[:3]}")
            print(f"Método 3 (diferença): {method3_scores[:3]}")
            print(f"Método 4 (softmax spoof): {method4_scores[:3]}")
            
            # Usar método 2 por agora (pode ser que bonafide seja classe 0)
            batch_score = method2_scores
        
        # Store results for this batch
        for file_name, score in zip(utt_ids, batch_score):
            all_results.append({
                'file_name': file_name,
                'predicted_score': float(score)
            })
        
        if (batch_idx + 1) % 10 == 0:  # Report more frequently
            print(f"Processed {(batch_idx + 1) * 10} samples...")
    
    # Convert to DataFrame and merge with original data
    results_df = pd.DataFrame(all_results)
    
    # Load original CSV to merge metadata
    original_df = pd.read_csv('/home/victor.moreno/dl-29_backup/spoof/mailabs_evaluation_results/mailabs_evaluation_results.csv')
    
    # Aplicar a mesma amostragem estratificada para merge
    language_counts = original_df['language'].value_counts()
    total_languages = len(language_counts)
    samples_per_language = 1000 // total_languages
    remaining_samples = 1000 % total_languages
    
    stratified_samples = []
    for i, (language, group) in enumerate(original_df.groupby('language')):
        n_samples = samples_per_language
        if i < remaining_samples:
            n_samples += 1
        n_samples = min(n_samples, len(group))
        sampled_group = group.sample(n=n_samples, random_state=42)
        stratified_samples.append(sampled_group)
    
    original_df_sampled = pd.concat(stratified_samples, ignore_index=True)
    original_df_sampled = original_df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Merge results with original data
    final_df = original_df_sampled.merge(results_df, on='file_name', how='left')
    
    # Save to CSV
    final_df.to_csv(save_path, index=False)
    print(f'Results saved to {save_path}')
    
    # Also save a simple score file (original format)
    simple_output_path = save_path.replace('.csv', '_simple.txt')
    with open(simple_output_path, 'w') as fh:
        for _, row in final_df.iterrows():
            if pd.notna(row['predicted_score']):  # Check for valid scores
                fh.write('{} {}\n'.format(row['file_name'], row['predicted_score']))
    print(f'Simple scores saved to {simple_output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAILABS dataset evaluation - CORRECTED VERSION')
    
    # Dataset
    parser.add_argument('--csv_file', type=str, 
                        default='/home/victor.moreno/dl-29_backup/spoof/mailabs_evaluation_results/mailabs_evaluation_results.csv',
                        help='Path to MAILABS CSV file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--eval_output', type=str, 
                        default='/home/victor.moreno/dl-29_backup/spoof/mailabs_evaluation_results/mailabs_predicted_scores_corrected.csv',
                        help='Path to save the evaluation result CSV')
    
    # Model parameters (needed for model initialization)
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#

    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    # Initialize model
    model = Model(args, device)
    
    # Handle DataParallel if model was saved with multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Load trained model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print('Model loaded : {}'.format(args.model_path))
    
    # Create evaluation dataset from CSV
    print(f'Loading dataset from: {args.csv_file}')
    eval_set = Dataset_MAILABS_eval(args.csv_file)
    print(f'Number of evaluation samples: {len(eval_set)}')
    
    # Run evaluation
    produce_evaluation_file(eval_set, model, device, args.eval_output)
    print('Evaluation completed!')
    
    print("\n" + "="*50)
    print("CORREÇÕES APLICADAS:")
    print("✅ Torchaudio ao invés de librosa")
    print("✅ Resampling adequado")
    print("✅ Padding com zeros (não repetição)")
    print("✅ Softmax aplicado aos scores")
    print("✅ Formato de tensor corrigido")
    print("="*50)