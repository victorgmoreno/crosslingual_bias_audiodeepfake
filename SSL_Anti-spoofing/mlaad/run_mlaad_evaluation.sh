#!/bin/bash

# Set paths
MLAAD_PATH="/home/victor.moreno/dl-29_backup/spoof/SSL_Anti-spoofing/mlaad"
OUTPUT_PATH="/home/victor.moreno/dl-29_backup/spoof/mlaad_evaluation_results"
MODEL_PATH="/home/victor.moreno/dl-29_backup/spoof/SSL_Anti-spoofing/models/model_weighted_CCE_100_14_1e-06_asvspoof5_training/epoch_9.pth"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --max_files_per_lang)
      MAX_FILES_PER_LANG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Step 1: Organize MLAAD dataset
echo "Step 1: Organizing MLAAD dataset..."
python3 $MLAAD_PATH/organize_mlaad.py \
  --mlaad_path "/home/victor.moreno/dl-29_backup/spoof/dataset/mlaad" \
  --output_path "$OUTPUT_PATH"

# Step 2: Evaluate ASVspoof model on MLAAD
echo "Step 2: Evaluating ASVspoof model on MLAAD..."
cd /home/victor.moreno/dl-29_backup/spoof/SSL_Anti-spoofing
python3 $MLAAD_PATH/evaluate_mlaad.py \
  --model_path "$MODEL_PATH" \
  --catalog_path "$OUTPUT_PATH/mlaad_catalog.json" \
  --output_path "$OUTPUT_PATH" \
  ${MAX_FILES_PER_LANG:+--max_files_per_lang $MAX_FILES_PER_LANG}

# Step 3: Analyze results
echo "Step 3: Analyzing results..."
if [ -f "$OUTPUT_PATH/evaluation_results.json" ]; then
  python3 $MLAAD_PATH/analyze_results.py \
    --results_path "$OUTPUT_PATH/evaluation_results.json" \
    --output_path "$OUTPUT_PATH/analysis"
else
  echo "Warning: evaluation_results.json not found. Skipping analysis step."
fi

echo "Evaluation completed. Results available in: $OUTPUT_PATH"