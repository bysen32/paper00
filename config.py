BATCH_SIZE = 40
LR = 0.01
WD = 1e-4
SAVE_FREQ = 1
INPUT_SIZE = (224, 224)
resume = ""
# resume = "checkpoint.ckpt"
test_model = "model.ckpt"
save_dir = "./save/"
DEV_MODE = True
TRAIN_CLASS = 20
VERSION_HEAD = "resnet50 model1 + model2 intra_class"

N_CLASSES = 8
N_SAMPLES = 6

# 20 x 2 91.1
# 10 x 4 90.9
# 8 x 5
# 5 x 8 91.8
# 4 x 10 90
# 2 x 20

# 20 x  2 91.8
# 10 x  4 91.7
#  8 x  5 92.2
#  5 x  8 90.7
#  4 x 10 88.7
#  2 x 20 13.8
