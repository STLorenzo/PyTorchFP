from pathlib import Path  # Path manipulation
import time  # Time measuring library
# Torch Libraries
import torch
# Personal Libraries
import src.ImageLoader as IL
import src.ImgConvNet as ICN
from src.general_functions import *

# DATA PREPROCESSING
VAL_PCT = 0.2
IMG_SIZE = (100, 100)
# TRAINING PARAMETERS
LR = 0.0001
BATCH_SIZE = 100
EPOCHS = 7
# DOC NAMES
MODEL_NAME = f"model-{int(time.time())}"
LOG_FILE = Path(f"../doc/{MODEL_NAME}.log")
# PATHS
p_conf_data = read_conf("/config/Project_conf.json")
BASE_DIR = Path(p_conf_data['base_path'])
DATA_BASE_DIR = BASE_DIR / p_conf_data['dirs']['data_dir']
IMG_DIR = DATA_BASE_DIR / "cats_dogs/PetImages"
PREDICT_DIR = DATA_BASE_DIR / "predictions"

# -------------------------------------Execution------------------------------------
REBUILD_DATA = False

img_loader = IL.ImageLoader()
if REBUILD_DATA:
    img_loader.make_training_data(val_pct=VAL_PCT)
#
MODEL_PATH = img_loader.created_data_path / "net_1.pl"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ICN.ImgConvNet(img_loader, DEVICE)
# net.optimize()
#
net.train_p(verbose=True, batch_size=BATCH_SIZE, max_epochs=EPOCHS, val_train_pct=0.25)
val_acc, val_loss = net.test_p(verbose=True)
print("Accuracy: ", val_acc)
print("Loss: ", val_loss)

# -------- PREDICTIONS ---------------
# net.make_predictions()

# -------- SAVE/LOAD ------------------

# net.save_net()
# net2 = ICN.ImgConvNet(img_loader, DEVICE)
# net2.load_net()
# net2.make_predictions()

# -------------- RESUME TRAINING --------------
#
# net.resume_training(DATA_BASE_DIR / "created_data/half_trained_models/__half__model-1590170392.0689793.pt")

# net.make_predictions()
