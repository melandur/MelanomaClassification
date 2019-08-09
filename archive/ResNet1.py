## training script for CIFAR10
import os, shutil, time
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorflow import tensorboard

from model import resnet_164

# CIFAR10_DIR = '/data/'

WORKERS = 1
BATCH_SIZE = 128
USE_CUDA = torch.cuda.is_available()
MAX_EPOCH = 150
PRINT_FREQUENCY = 100

if USE_CUDA:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

# load data


class ResNet():


# if not os.path.exists(CIFAR10_DIR):
#     raise RuntimeError('Cannot find CIFAR10 directory')

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
# train_set = CIFAR10(root=CIFAR10_DIR, train=True, transform=transforms.Compose([
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomCrop((32, 32), 4),
#                         transforms.ToTensor(), normalize]))
#
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
#
# val_loader = DataLoader(CIFAR10(root=CIFAR10_DIR, train=False, transform=transforms.Compose([
#                                                         transforms.ToTensor(), normalize])),
#                                                         batch_size=BATCH_SIZE, shuffle=False,
#                                                         num_workers=WORKERS, pin_memory=True)




    # get resnet-164
    def get_model():
        model = resnet_164(output_classes=3)
        if USE_CUDA:
            model = model.cuda()
        return model


    # remove existing log directory
    def remove_log():
        if os.path.exists('./log'):
            shutil.rmtree('./log')
            os.mkdir('./log')


# Metric
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# top-k accuracy
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# validation
def validate(model, ceriterion):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for ind, (x, label) in enumerate(val_loader):
        if USE_CUDA:
            x, label = x.cuda(), label.cuda()
        vx, vl = Variable(x, volatile=True), Variable(label, volatile=True)

        score = model(vx)
        loss = ceriterion(score, vl)
        prec1 = accuracy(score.data, label)

        losses.update(loss.data[0], x.size(0))
        top1.update(prec1[0][0], x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: [{0}/{0}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
          len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    return top1.avg, losses.avg


# train
def train(model):
    remove_log()
    writer = tensorboard.SummaryWriter('./log')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=0.0001)
    ceriterion = nn.CrossEntropyLoss()
    step = 1
    for epoch in range(1, MAX_EPOCH + 1):
        if epoch == 80 or epoch == 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model.train()
        end = time.time()

        for ind, (x, label) in enumerate(train_loader):
            data_time.update(time.time()-end)
            if USE_CUDA:
                x, label = x.cuda(), label.cuda()
            vx, vl = Variable(x), Variable(label)

            score = model(vx)
            loss = ceriterion(score, vl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            batch_time.update(time.time()-end)
            prec1 = accuracy(score.data, label)

            losses.update(loss.data[0], x.size(0))
            top1.update(prec1[0][0], x.size(0))

            writer.add_scalar('train_loss', loss.data[0], step)
            writer.add_scalar('train_acc', prec1[0][0], step)

            if (ind+1) % PRINT_FREQUENCY == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, ind+1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            end = time.time()
        top1, test_loss = validate(model, ceriterion)
        writer.add_scalar('test_loss', test_loss, step)
        writer.add_scalar('test_acc', top1, step)

        if epoch % 30 == 0:
            torch.save({'state_dcit': model.state_dict(),
                        'accuracy': top1},
                        'epoch-{:03d}-model.pth.tar'.format(epoch))


if __name__ == '__main__':
    model = get_model()
    train(model)