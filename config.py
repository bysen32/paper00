BATCH_SIZE = 40
LR = 0.01
WD = 1e-4
SAVE_FREQ = 1
INPUT_SIZE = (224, 224)
resume = ""
# resume = "checkpoint.ckpt"
test_model = "model.ckpt"
save_dir = "./save/"
DEV_MODE = False
TRAIN_CLASS = 200
TRAIN_SAMPLE = 5994
# TRAIN_SAMPLE = 600
VERSION_HEAD = "resnet50 model1 + model2 intra_class"

N_CLASSES = 10
N_SAMPLES = 10