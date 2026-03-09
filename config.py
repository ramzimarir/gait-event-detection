"""Project configuration for gait analysis."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DATA_DIR = DATA_DIR
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_WEIGHTS_DIR = OUTPUT_DIR / "models"

FS = 100
MODEL_IN_CHANNELS = 10

TIME_COLUMN = "time"

WINDOW_SIZE = 200
OVERLAP = 0.5

FEATURE_COLUMNS = [
    "x_acc_left",
    "y_acc_left",
    "z_acc_left",
    "x_acc_right",
    "y_acc_right",
    "z_acc_right",
    "x_gyro_left",
    "y_gyro_left",
    "z_gyro_left",
    "x_gyro_right",
    "y_gyro_right",
    "z_gyro_right",
    "quat_1_left",
    "quat_2_left",
    "quat_3_left",
    "quat_4_left",
    "quat_1_right",
    "quat_2_right",
    "quat_3_right",
    "quat_4_right",
]

LABEL_COLUMNS = [
    "TO_left",
    "IC_left",
    "TO_right",
    "IC_right",
]
