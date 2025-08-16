# MLAAD Evaluation Framework

A clean, efficient framework for evaluating audio anti-spoofing models on the Multi-Language Audio Anti-Spoofing Dataset (MLAAD).

## Overview

This framework evaluates how well an ASVspoof model generalizes across different languages and text-to-speech architectures in the MLAAD dataset.

## Features

- Evaluate a pre-trained ASVspoof model on MLAAD dataset
- Filter evaluation by specific language
- Limit the number of files per model for quicker testing
- Generate detailed statistics by language and TTS architecture
- Create visualizations of model performance

## Requirements

- Python 3.8+
- PyTorch
- librosa
- pandas
- matplotlib
- seaborn
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the evaluation with default settings:

```bash
python main.py
```

This will:
- Use the model at the default path
- Evaluate on the full MLAAD dataset
- Save results to the default output directory

### Custom Usage

```bash
# Evaluate only on English
python main.py --language en

# Limit to 100 files per model (for faster testing)
python main.py --max_files 100

# Specify custom paths
python main.py --model_path /path/to/model.pth --mlaad_path /path/to/mlaad --output_dir /path/to/output

# Use a different batch size
python main.py --batch_size 64
```

## Output

The evaluation produces the following output:

- `results.json`: Raw evaluation results for each audio file
- `language_stats.csv`: Performance statistics by language
- `architecture_stats.csv`: Performance statistics by TTS architecture
- Visualizations:
  - `language_scores.png`: Average scores by language
  - `architecture_scores.png`: Average scores by architecture
  - `score_distribution.png`: Distribution of all scores
  - `language_high_conf.png`: High confidence ratio by language

## Directory Structure

```
mlaad_eval/
├── config.py              # Configuration and default paths
├── data.py                # Dataset handling and loading
├── evaluate.py            # Core evaluation logic
├── visualize.py           # Visualization and reporting functions
├── main.py                # Main entry point script
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Customization

You can modify the default paths and configuration parameters in `config.py`.

## Performance Notes

- For large datasets, consider using a smaller value for `--max_files`
- Use a smaller batch size if you encounter memory issues
- The evaluation will automatically use CUDA if available