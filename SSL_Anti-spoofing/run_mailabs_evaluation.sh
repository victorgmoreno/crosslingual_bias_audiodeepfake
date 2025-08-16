#!/bin/bash
#SBATCH --job-name=mailabs_eval
#SBATCH --nodelist=dl-01
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=04:00:00

# Directories
MODEL_PATH="/home/victor.moreno/dl-29_backup/spoof/SSL_Anti-spoofing/models/model_weighted_CCE_100_14_1e-06_asvspoof5_training/epoch_9.pth"
CSV_PATH="/home/victor.moreno/dl-29_backup/spoof/dataset/code/mailabs_unified_meta.csv"
OUTPUT_PATH="/home/victor.moreno/dl-29_backup/spoof/dataset/code/mailabs_evaluation_results.csv"
ANALYSIS_DIR="/home/victor.moreno/dl-29_backup/spoof/dataset/code/analysis_results"

# Ensure the analysis directory exists
mkdir -p $ANALYSIS_DIR

echo "Starting evaluation of anti-spoofing model on M-AILABS dataset..."
python evaluate_mailabs.py \
    --model_path $MODEL_PATH \
    --csv_path $CSV_PATH \
    --output_path $OUTPUT_PATH \
    --batch_size 64 \
    --num_workers 16 \
    --threshold 0.5 \
    --world_size 4

echo "Evaluation complete. Starting analysis..."
python analyze_results.py \
    --results_csv $OUTPUT_PATH \
    --output_dir $ANALYSIS_DIR

echo "Analysis complete. Results saved to $ANALYSIS_DIR"