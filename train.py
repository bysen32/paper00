import os
import sys
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from datetime import datetime
from config import resume, save_dir, parse_option
import core
from core import model, dataset, resnet, visible
from core.utils import init_log, progress_bar, AverageMeter, adjust_learning_rate, warmup_learning_rate, accuracy
# from torch.utils.tensorboard import SummaryWriter
from core.losses import SupConLoss, PairPairSupConLoss


def set_model(opt):
    model = core.model.MyNet()
    criterion = PairPairSupConLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def set_loader(opt):
    if opt.dataset == 'cub200':
        train_dataset = dataset.BatchDataset(root=opt.data_folder)
        train_sampler = dataset.BalancedBatchSampler(
            train_dataset, opt.n_classes, opt.n_samples)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

        val_dataset = dataset.CUB_Test(root=opt.data_folder)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, drop_last=False)

        return train_loader, val_loader
    elif opt_dataset == 'dog':
        pass


def set_optimizer(opt, model):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    return optimizer


def train(train_loader, model, criterion_encoder, optimizer_encoder, classifier, criterion_fc, optimizer_fc, epoch, opt):
    model.train()

    losses = AverageMeter()
    fc_losses = AverageMeter()
    encoder_losses = AverageMeter()
    top1 = AverageMeter()

    _print("--" * 50)
    for i, (images, labels) in enumerate(train_loader):
        images = torch.cat(images, dim=0).cuda()
        labels = torch.cat([labels, labels], dim=0).cuda()
        BSZ = labels.shape[0]

        # warmup_learning_rate(opt, epoch, i, len(train_loader), optimizer_encoder)
        # warmup_learning_rate(opt, epoch, i, len(train_loader), optimizer_fc)

        features = model(images, labels)
        # encoder_loss = criterion_encoder(features, labels)
        # encoder_losses.update(encoder_loss.item(), BSZ)

        logits = classifier(features)
        fc_loss = criterion_fc(logits, labels)
        fc_losses.update(fc_loss.item(), BSZ)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        top1.update(acc1[0], BSZ)

        # loss = encoder_loss + fc_loss
        loss = fc_loss
        losses.update(loss.item(), BSZ)

        optimizer_encoder.zero_grad()
        optimizer_fc.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_fc.step()

        progress_bar(i, len(train_loader), "train")

    _print("epoch:{}".format(epoch))
    _print(
        "loss: {loss.avg:.4f} "
        "encoder_loss: {encoder_loss.avg:.4f} "
        "fc_loss: {fc_loss.avg:.4f} "
        "train acc@1 {top1.avg:.4f} ".format(
            loss=losses, encoder_loss=encoder_losses, fc_loss=fc_losses, top1=top1))
    return losses.avg, top1.avg


# evaluate on test set
def validate(val_loader, model, criterion_encoder, classifier, criterion_fc, epoch, opt):
    model.eval()

    losses = AverageMeter()
    fc_losses = AverageMeter()
    encoder_losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()
            BSZ = labels.shape[0]

            # caculate class loss
            features = model(images, labels)
            # encoder_loss = criterion_encoder(features, labels)
            # encoder_losses.update(encoder_loss.item(), BSZ)

            logits = classifier(features)
            fc_loss = criterion_fc(logits, labels)
            fc_losses.update(fc_loss.item(), BSZ)

            loss = fc_loss
            losses.update(loss.item(), BSZ)

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1[0], BSZ)

            progress_bar(i, len(val_loader), "eval test set")

    _print(
        "loss: {loss.avg:.4f} "
        "fc_loss :{fc_loss.avg:.4f} "
        "acc@1 {top1.avg:.4f}".format(loss=losses, fc_loss=fc_losses, top1=top1))

    return losses.avg, top1.avg


def main():
    cudnn.benchmark = True

    opt = parse_option()

    train_loader, val_loader = set_loader(opt)

    model, criterion_encoder = set_model(opt)

    classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(2048, 200)
    ).cuda()
    criterion_fc = torch.nn.CrossEntropyLoss().cuda()

    optimizer_encoder = set_optimizer(opt, model)
    optimizer_fc = set_optimizer(opt, classifier)

    best_acc = 0
    for epoch in range(opt.epochs):
        adjust_learning_rate(opt, optimizer_encoder, epoch)
        adjust_learning_rate(opt, optimizer_fc, epoch)

        loss, train_acc = train(train_loader, model, criterion_encoder, optimizer_encoder,
                                classifier, criterion_fc, optimizer_fc, epoch, opt)

        loss, val_acc = validate(
            val_loader, model, criterion_encoder, classifier, criterion_fc, epoch, opt)

        best_acc = max(best_acc, val_acc)
        _print("best accuracy:{:.2f}".format(best_acc))


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 存取模型代码
# start_epoch = 1

save_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
if os.path.exists(save_dir):
    raise NameError("model dir exists!")
os.makedirs(save_dir)

logging = init_log(save_dir)
_print = logging.info

# load saved model
# if resume and os.path.isfile(resume):
#     ckpt = torch.load(resume)
#     model.load_state_dict(ckpt["net_state_dict"])
#     start_epoch = ckpt["epoch"] + 1
#     _print("Load {}".format(resume))

if __name__ == "__main__":
    main()
    print("finishing training")
