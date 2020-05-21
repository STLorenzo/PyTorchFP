from pathlib import Path  # Path manipulation
import time  # Time measuring library
# Torch Libraries
import torch
# Personal Libraries
import ImageLoader as IL
import ImgConvNet as ICN

# DATA PREPROCESSING
VAL_PCT = 0.2
IMG_SIZE = (50, 50)
# TRAINING PARAMETERS
LR = 0.001
BATCH_SIZE = 32
EPOCHS = 20
# DOC NAMES
MODEL_NAME = f"model-{int(time.time())}"
LOG_FILE = Path(f"../doc/{MODEL_NAME}.log")
# PATHS
DATA_BASE_DIR = Path("../data")
IMG_DIR = DATA_BASE_DIR / "cats_dogs/PetImages"
PREDICT_DIR = DATA_BASE_DIR / "predictions"

# -------------------------------------Execution------------------------------------
REBUILD_DATA = False

img_loader = IL.ImageLoader(DATA_BASE_DIR, IMG_DIR, PREDICT_DIR, IMG_SIZE)
if REBUILD_DATA:
    img_loader.make_training_data(val_pct=VAL_PCT)

MODEL_PATH = img_loader.created_data_path / "net_1.pl"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ICN.ImgConvNet(img_loader, DEVICE)
# net.optimize()

# net.train_p(verbose=True, batch_size=BATCH_SIZE, max_epochs=EPOCHS)
# val_acc, val_loss = net.test_p(verbose=True)
# print("Accuracy: ", val_acc)
# print("Loss: ", val_loss)

# -------- PREDICTIONS ---------------
# net.make_predictions()

# -------- SAVE/LOAD ------------------

# net.save_net(MODEL_PATH)
# net2 = ImgConvNet(img_loader, DEVICE)
# net2.load_net(MODEL_PATH)
# net2.make_predictions()

# -------------- RESUME TRAINING --------------
net.resume_training(DATA_BASE_DIR / "created_data/__half__model-1590075563.4901638.pt")