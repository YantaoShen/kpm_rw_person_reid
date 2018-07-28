from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable


from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer,  RandomWalkGrpShufTrainer
from reid.evaluators import Evaluator, CascadeEvaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler, RandomMultipleGallerySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.models.embedding import RandomWalkEmbed
from reid.models.multi_branch import  RandomWalkKpmNet


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomSizedEarser(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomMultipleGallerySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (384, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    base_model = models.create(args.arch, num_features=1024, cut_at_pooling=True,
                          dropout=args.dropout, num_classes=args.features)

    grp_num = args.grp_num
    embed_model = [RandomWalkEmbed(instances_num=args.num_instances,
                            feat_num=(2048 / grp_num), num_classes=2,
                            drop_ratio=args.dropout).cuda() for i in range(grp_num)]

    base_model = nn.DataParallel(base_model).cuda()

    model = RandomWalkKpmNet(instances_num=args.num_instances,
                        base_model=base_model, embed_model=embed_model, alpha=args.alpha)
    
    if args.retrain:
        if args.evaluate_from:
            print('loading trained model...')
            checkpoint = load_checkpoint(args.evaluate_from)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('loading base part of pretrained model...')
            checkpoint = load_checkpoint(args.retrain)
            #copy_state_dict(checkpoint['state_dict'], base_model, strip='base.module.', replace='module.')
            copy_state_dict(checkpoint['state_dict'], base_model, strip='base_model.', replace='')
            print('loading embed part of pretrained model...')
            if grp_num > 1:
                for i in range(grp_num):
                    copy_state_dict(checkpoint['state_dict'], embed_model[i], strip='embed_model.bn_'+str(i)+'.', replace='bn.')
                    copy_state_dict(checkpoint['state_dict'], embed_model[i], strip='embed_model.classifier_'+str(i)+'.', replace='classifier.')
            else:
                copy_state_dict(checkpoint['state_dict'], embed_model[0], strip='module.embed_model.', replace='')

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

        # Load from checkpoint
    start_epoch = best_top1 = 0
    best_mAP = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    # Evaluator
    evaluator = CascadeEvaluator(
                            base_model,
                            embed_model,
                            embed_dist_fn=lambda x: F.softmax(x, dim=1).data[:, 0])

    if args.evaluate:
        metric.train(model, train_loader)
        if args.evaluate_from:
            print('loading trained model...')
            checkpoint = load_checkpoint(args.evaluate_from)
            model.load_state_dict(checkpoint['state_dict'])
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, args.alpha, metric, rerank_topk=args.rerank, dataset=args.dataset)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # base lr rate and embed lr rate
    new_params = [z for z in model.embed]
    param_groups = [
        {'params': model.base.module.base.parameters(), 'lr_mult': 1.0}] + \
        [{'params': new_params[i].parameters(), 'lr_mult': 10.0} for i in range(grp_num)]

    # Optimizer
    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Trainer
    trainer = RandomWalkGrpShufTrainer(model, criterion, args.alpha, grp_num)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = args.ss if args.arch == 'inception' else 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        return lr

    # Start training
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, lr, warm_up=False)
        top1, mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, args.alpha, second_stage=True, dataset=args.dataset)

        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, args.alpha, metric, rerank_topk=args.rerank, dataset=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--grp-num', type=int, default=1)
    parser.add_argument('--rerank', type=int, default=75)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--ss', type=int, default=40,
                        help="step size for adjusting learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--retrain', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--evaluate-from', type=str, default='', metavar='PATH')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
