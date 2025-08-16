"""
Configuration parameters for MLAAD evaluation.
"""

# Default paths
MODEL_PATH = '/home/victor.moreno/dl-29_backup/spoof/SSL_Anti-spoofing/models/model_weighted_CCE_100_14_1e-06_asvspoof5_training/epoch_9.pth'
MLAAD_PATH = '/home/victor.moreno/dl-29_backup/spoof/dataset/mlaad'
OUTPUT_DIR = '/home/victor.moreno/dl-29_backup/spoof/mlaad_results'

# SSL-Antispoofing model path
SSL_ANTISPOOFING_PATH = '/home/victor.moreno/dl-29_backup/spoof/SSL_Anti-spoofing'

# Model configuration
BATCH_SIZE = 32
NUM_WORKERS = 4
AUDIO_CUT_LENGTH = 64600  # ~4 seconds at 16kHz
SAMPLE_RATE = 16000