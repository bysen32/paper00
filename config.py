import argparse

BATCH_SIZE = 40
SAVE_FREQ = PRINT_FREQ = 1
INPUT_SIZE = (224, 224)

LEARNING_RATE = 0.01  # 0.05
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

resume = ""
# resume = "checkpoint.ckpt"
test_model = "model.ckpt"
save_dir = "./save/"

DEV_MODE = True
DATA_FOLDER = "./CUB_200_2011"
TRAIN_CLASS = 20
# TRAIN_SAMPLE = 5994
TRAIN_SAMPLE = 600

N_CLASSES = 8
N_SAMPLES = 5


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=PRINT_FREQ)
    parser.add_argument("--save_freq", type=int, default=SAVE_FREQ)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--devmode", type=int, default=DEV_MODE)

    # optimization
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr_decay_epochs", type=str,
                        default='20,70', help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float,
                        default=0.1, help="decay rate for learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)

    # model
    parser.add_argument("--model", type=str, default="resnet50")

    # dataset
    parser.add_argument("--dataset", type=str, default="cub200",
                        choices=["cub200", "stanford_dog"])
    parser.add_argument('--mean', type=str,
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str,
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--n_classes', type=int, default=N_CLASSES)
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES)
    parser.add_argument("--data_folder", type=str,
                        default=DATA_FOLDER, help="path to custom dataset")

    # method
    parser.add_argument("--method", type=str,
                        default="PPSupCon", choices=["PPSupCon", "SupCon"])

    # temperature
    parser.add_argument("--temp", type=float, default=0.07,
                        help="temperature for loss function")

    # other setting
    parser.add_argument("--cosine", action="store_true",
                        help="using consine annealing")
    parser.add_argument("--syncBN", action="store_true",
                        help="using synchronized batch nomalization")
    parser.add_argument("--warm", action="store_true",
                        help="warm-up for large batch training")
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.batch_size > 256:
        opt.warm = True
    opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    # if not os.path.isdir(opt.tb_folder):
    #     os.makedirs(opt.tb_folder)

    # opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    # if not os.path.isdir(opt.save_folder):
    #     os.makedirs(opt.save_folder)

    return opt
