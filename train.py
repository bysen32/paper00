import os
import sys
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from config import BATCH_SIZE, SAVE_FREQ, LR, resume, save_dir, WD, DEV_MODE, VERSION_HEAD, N_CLASSES, N_SAMPLES, TRAIN_CLASS
from core import model, dataset, resnet, visible
from core.utils import init_log, progress_bar, AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
start_epoch = 1
best_acc = 0
save_dir = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
if os.path.exists(save_dir):
    raise NameError("model dir exists!")

os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info

# read dataset
# trainset = dataset.CUB(root="./CUB_200_2011", is_train=True, data_len=None)
# trainset = dataset.CUB_Train(root="./CUB_200_2011")
trainset = dataset.BatchDataset(root="./CUB_200_2011")
train_sampler = dataset.BalancedBatchSampler(trainset, N_CLASSES, N_SAMPLES)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

valset = dataset.CUB_Val(root="./CUB_200_2011")
valloader = torch.utils.data.DataLoader(
    valset, batch_size=100, shuffle=False, num_workers=8, drop_last=False)

# testset = dataset.CUB(root="./CUB_200_2011", is_train=False, data_len=None)
testset = dataset.CUB_Test(root="./CUB_200_2011")
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=8, drop_last=False)

# define model
net = model.MyNet()
# net.avgpool = torch.nn.AdaptiveAvgPool2d(1)
# net.fc = torch.nn.Linear(512 * 4, 200)

# load saved model
if resume and os.path.isfile(resume):
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt["net_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    _print("Load {}".format(resume))

criterion = torch.nn.CrossEntropyLoss().cuda()
rank_criterion = torch.nn.MarginRankingLoss(margin=0.05)
softmax_layer = torch.nn.Softmax(dim=1).cuda()

# define optimizers
raw_parameters = list(net.parameters())
raw_optimizer = torch.optim.SGD(
    raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)

schedulers = [
    MultiStepLR(raw_optimizer, milestones=[50, 100], gamma=0.1),
    # CosineAnnealingLR(raw_optimizer, 100 * len(trainloader))
]

# ----------------- pairs struct -----------------
# rank_criterion = torch.nn.MarginRankingLoss(margin=0.05)
# softmax_layer = torch.nn.Softmax(dim=1).cuda()

net = net.cuda()
net = DataParallel(net)

_print("-"*10+VERSION_HEAD+"-"*10)

waTestIndex = torch.zeros(len(testset))

for epoch in range(start_epoch, 500):

    _print("--" * 50)
    # _print("resnet50 model1: cossim model2 224x224 batchsize:30")
    net.train()
    for i, (images, labels, idxs) in enumerate(trainloader):
        images = torch.cat(images, dim=0).cuda()
        labels = torch.cat([labels, labels], dim=0).cuda()
        idxs = torch.cat([idxs, idxs], dim=0).cuda()

        raw_optimizer.zero_grad()

        raw_logits, _, raw_features, logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2 = net(
            images, labels, idxs)
        batch_size = logit1_self.shape[0]

        self_logits = torch.zeros(2*batch_size, 200).cuda()
        other_logits = torch.zeros(2*batch_size, 200).cuda()
        self_logits[:batch_size] = logit1_self
        self_logits[batch_size:] = logit2_self
        other_logits[:batch_size] = logit1_other
        other_logits[batch_size:] = logit2_other

        # logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2], dim=0)
        softmax_loss = criterion(self_logits, targets)

        raw_loss = criterion(raw_logits, labels)

        self_scores = softmax_layer(self_logits)[torch.arange(
            2*batch_size).cuda().long(), torch.cat([labels1, labels2], dim=0)]
        other_scores = softmax_layer(other_logits)[torch.arange(
            2*batch_size).cuda().long(), torch.cat([labels1, labels2], dim=0)]
        flag = torch.ones([2 * batch_size, ]).cuda()
        rank_loss = rank_criterion(self_scores, other_scores, flag)

        # raw1_self_loss = criterion(logit1_self, labels1)
        # raw2_self_loss = criterion(logit2_self, labels2)
        # feature_loss = criterion(features, labels)

        # target = torch.autograd.Variable(torch.ones(batch_size, 1)).cuda()
        # intra_dist_loss = torch.nn.CosineEmbeddingLoss(reduction="mean")(projected_features[:batch_size], projected_features[batch_size:], target)
        # dist_loss = torch.nn.TripletMarginLoss()(inter_pairs[0], intra_pairs[1], inter_pairs[1])

        # ---------- pairs attention struct ---------------------
        # total_loss = raw_loss + dist_loss + intra_dist_loss
        # total_loss = raw_loss + raw1_self_loss + raw2_self_loss
        # total_loss = raw1_self_loss + raw2_self_loss
        # total_loss = raw_loss
        total_loss = softmax_loss + raw_loss + rank_loss
        total_loss.backward()

        raw_optimizer.step()

        progress_bar(i, len(trainloader), "train")

    for sch in schedulers:
        sch.step()

    model.trainend()

    if epoch % SAVE_FREQ == 0:
        raw_losses = AverageMeter()
        raw1_losses = AverageMeter()
        raw2_losses = AverageMeter()
        features_loss_total = 0
        dist_losses = AverageMeter()
        intra_dist_losses = AverageMeter()
        inter_dist_losses = AverageMeter()
        g_WALabelCount = torch.zeros(TRAIN_CLASS)
        g_TrainLabelCount = torch.zeros(TRAIN_CLASS)
        train_correct = 0
        total = 0
        net.eval()
        for i, (images, labels, idxs) in enumerate(trainloader):
            with torch.no_grad():
                images = torch.cat(images, dim=0)
                labels = torch.cat([labels, labels], dim=0).cuda()
                idxs = torch.cat([idxs, idxs], dim=0).cuda()
                batch_size = images.size(0)

                raw_logits, _, raw_features, logits1_self, logits2_other, logits2_self, logits2_other, labels1, labels2 = net(
                    images, labels, idxs)
                # caculate class loss
                raw_loss = criterion(raw_logits, labels)
                # raw1_self_loss = criterion(logits1_self, labels1)
                # raw2_self_loss = criterion(logits2_self, labels2)
                # features_loss = criterion(features, labels)

                # target = torch.autograd.Variable(torch.ones(batch_size, 1)).cuda()
                # intra_dist_loss = torch.nn.CosineEmbeddingLoss(reduction="mean")( projected_features[:batch_size], projected_features[batch_size:], target)
                # inter_dist_loss = torch.nn.CosineEmbeddingLoss(reduction="mean")(inter_pairs[0], inter_pairs[1], target)
                # flag = torch.ones(batch_size, 1).cuda()
                # inter_dist_loss = torch.nn.MarginRankingLoss(margin=0.05)(
                # inter_pairs[0], inter_pairs[1], flag)
                # dist_loss = torch.nn.TripletMarginLoss()(projected_features[:batch_size], projected_features[batch_size:], inter_pairs[1])
                # dist_loss = torch.nn.TripletMarginLoss()(inter_pairs[0], intra_pairs[1], inter_pairs[1])

                # visible.plot_embedding(
                #    raw_features, torch.cat([labels, labels], dim=0), "raw_feature")

                # caculate accuracy
                _, raw_predict = torch.max(raw_logits, 1)
                total += batch_size

                res = raw_predict.data == labels.data
                # waIdxs = torch.where(res == False)
                # waLabels = torch.cat([labels, labels], dim=0)[waIdxs]
                # g_WALabelCount[waLabels] += 1
                # g_TrainLabelCount[labels] += 2

                train_correct += torch.sum(raw_predict.data == labels.data)
                # features_loss_total += features_loss
                raw_losses.update(raw_loss.item(), batch_size)
                # raw1_losses.update(raw1_self_loss.item(), batch_size)
                # raw2_losses.update(raw2_self_loss.item(), batch_size)
                # dist_losses.update(dist_loss.item(), batch_size)
                # intra_dist_losses.update(intra_dist_loss.item(), batch_size)
                # inter_dist_losses.update(inter_dist_loss.item(), batch_size)
                progress_bar(i, len(trainloader), "eval train set")

        train_acc = float(train_correct) / total
        # train_loss = (raw_losses.avg + dist_losses.avg + intra_dist_losses.avg + inter_dist_losses.avg)
        train_loss = (raw_losses.avg)

        # _print("epoch:{} - train loss: {:.3f} raw_loss: {:.3f} dist_loss: {:.3f} intra_loss: {:.3f} inter_loss: {:.3f}".format(epoch, train_loss, raw_losses.avg, dist_losses.avg, intra_dist_losses.avg, inter_dist_losses.avg))
        # _print("epoch:{} - train loss: {:.3f} raw_loss: {:.3f} dist_loss: {:.3f} intra_loss: {:.3f} inter_loss: {:.3f}".format(epoch, train_loss, raw_losses.avg, dist_losses.avg, intra_dist_losses.avg, inter_dist_losses.avg))
        _print("train acc: {:.3f} total sample: {}".format(train_acc, total))
        # _print(g_WALabelCount/g_TrainLabelCount)

        # evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, labels, index = data[0].cuda(), data[1].cuda(), data[2]
                batch_size = img.size(0)

                raw_logits = net(img, labels, flag="test")
                # caculate class loss
                raw_loss = criterion(raw_logits, labels)

                # caculate accuracy
                _, raw_predict = torch.max(raw_logits, 1)
                total += batch_size
                res = (raw_predict.data == labels.data)
                # _print(torch.where(res == False))
                #_print(index[torch.where(res == False)])
                # waTestIndex[index[torch.where(res == False)]] += 1

                test_correct += torch.sum(raw_predict.data == labels.data)
                test_loss += raw_loss.item() * batch_size
                progress_bar(i, len(testloader), "eval test set")

        test_acc = float(test_correct) / total
        if (best_acc < test_acc):
            best_acc = test_acc
        test_loss = test_loss / total
        _print("epoch:{} - test loss: {:.4f}".format(epoch, test_loss))

        _print(
            "test acc: {:.3f}/{:.3f} total sample: {}".format(test_acc, best_acc, total))

        # _print(waTestIndex/epoch)

        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # 暂时不保存模型
        torch.save(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "net_state_dict": net_state_dict,
            },
            "checkpoint.ckpt"
        )

print("finishing training")
