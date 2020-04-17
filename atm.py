from __future__ import print_function, absolute_import
from reid.snatch import *
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os

from reid.utils.logging import Logger
import os.path as osp
import sys
from torch.backends import cudnn
from reid.utils.serialization import load_checkpoint
from torch import nn
import time
import math
import pickle
import time

import matplotlib.pyplot as plt

import os
import codecs
from common_tool import *


def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    one_shot, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))
    l_data = []
    total_step = args.total_step
    mv_num = math.ceil(len(u_data) / total_step)  # 最后一轮必定不足add_num的数量
    tagper_num = math.ceil(len(u_data) / args.train_tagper_step)
    # 输出该轮训练关键的提示信息
    print(
        "{} training begin with dataset:{},batch_size:{},epoch:{},step_size:{},max_frames:{},total_step:{},add {} sample each step.".format(
            args.exp_name, args.dataset, args.batch_size, args.epoch, args.step_size, args.max_frames, total_step,
            mv_num))

    # 指定输出文件
    # 第三部分要说明关键参数的设定
    reid_path = osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order)
    sys.stdout = Logger(osp.join(reid_path, 'log' + time.strftime(".%m_%d_%H-%M-%S") + '.txt'))
    data_file = codecs.open(osp.join(reid_path, 'data.txt'), mode='a')
    time_file = codecs.open(osp.join(reid_path, 'time.txt'), mode='a')
    # save_path = reid_path
    tagper_path = osp.join(reid_path, 'tagper')
    tagper_file = codecs.open(osp.join(reid_path, "tagper_data.txt"), mode='a')

    # resume_step, ckpt_file = -1, ''
    # if args.resume:  # 重新训练的时候用
    #     resume_step, ckpt_file = resume(args)

    # initial the EUG algorithm
    reid = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids,
               data_dir=dataset_all.images_dir, l_data=one_shot, u_data=u_data, save_path=reid_path,
               max_frames=args.max_frames)
    tagper = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode,
                 num_classes=dataset_all.num_train_ids,
                 data_dir=dataset_all.images_dir, l_data=one_shot, u_data=u_data, save_path=reid_path,
                 max_frames=args.max_frames)
    # 开始的时间记录
    exp_start = time.time()
    for step in range(total_step):

        print("{} training begin with dataset:{},batch_size:{},epoch:{},step:{}/{} saved to {}.".format(args.exp_name,
                                                                                                        args.dataset,
                                                                                                        args.batch_size,
                                                                                                        args.epoch,
                                                                                                        step + 1,
                                                                                                        total_step,
                                                                                                        reid_path))
        print("key parameters contain mv_num:{} tagper_num:{} len(l_data):{},len(u_data):{}".format(mv_num, tagper_num,
                                                                                                    len(l_data),
                                                                                                    len(u_data)))

        # 开始训练
        train_reid_data = one_shot + l_data  # 在这个过程中,保持了one_shot不变了
        reid_start = time.time()
        if step == 0 and not args.is_baseline:
            reid.resume(osp.join(reid_path, 'Dissimilarity_step_0.ckpt'), 0)
        else:
            reid.train(train_reid_data, step, tagper=0, epochs=args.epoch, step_size=args.step_size, init_lr=0.1)

        # 开始评估
        # mAP, top1, top5, top10, top20 =0,0,0,0,0
        mAP, top1, top5, top10, top20 = reid.evaluate(dataset_all.query, dataset_all.gallery)
        # 测试 train tagper之前的select_pre
        pred_y, pred_score, label_pre = reid.estimate_label_atm3(u_data, l_data, one_shot)  # 针对u_data进行标签估计
        selected_idx = reid.select_top_data(pred_score, min(mv_num * (step + 1), len(u_data)))
        select_pre = reid.get_select_pre(selected_idx, pred_y, u_data)

        reid_end = time.time()
        data_file.write(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} len(l_data):{} label_pre:{:.2%} select_pre:{:.2%}\n".format(
                int(step + 1), mAP, top1, top5, top10, top20, len(l_data), label_pre, select_pre))
        print(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} len(l_data):{} label_pre:{:.2%} select_pre:{:.2%} \n".format(
                int(step + 1), mAP, top1, top5, top10, top20, len(l_data), label_pre, select_pre))

        tagper_start = time.time()
        '''第一个tagper可以resume'''
        # if step == 0  and not args.is_baseline:
        #     tagper.resume(osp.join(reid_path,'tagper','Dissimilarity_step_0.ckpt'), 0)
        # else:
        #     tagper.resume(osp.join(reid_path, 'Dissimilarity_step_{}.ckpt'.format(step)), step)
        #     selected_idx = tagper.select_top_data(pred_score, min(tagper_num*(step+1),len(u_data)))  #训练tagper的数量也递增
        #     new_train_data = tagper.generate_new_train_data_only(selected_idx, pred_y, u_data)  # 这个选择准确率应该是和前面的label_pre是一样的.
        #     train_tagper_data = one_shot+l_data+new_train_data
        #     tagper.train(train_tagper_data, step, tagper=1, epochs=args.epoch, step_size=args.step_size, init_lr=0.1)
        '''所有的tagper都重新训练'''
        tagper.resume(osp.join(reid_path, 'Dissimilarity_step_{}.ckpt'.format(step)), step)
        selected_idx = tagper.select_top_data(pred_score, min(tagper_num * (step + 1), len(u_data)))  # 训练tagper的数量也递增
        new_train_data = tagper.generate_new_train_data_only(selected_idx, pred_y,
                                                             u_data)  # 这个选择准确率应该是和前面的label_pre是一样的.
        train_tagper_data = one_shot + new_train_data
        tagper.train(train_tagper_data, step, tagper=1, epochs=args.epoch, step_size=args.step_size, init_lr=0.1)

        # 开始评估
        # mAP, top1, top5, top10, top20 =0,0,0,0,0
        mAP, top1, top5, top10, top20 = tagper.evaluate(dataset_all.query, dataset_all.gallery)
        pred_y, pred_score, label_pre = tagper.estimate_label_atm3(u_data, l_data, one_shot)

        # 下面正对 reid 移动数据.
        selected_idx = tagper.select_top_data(pred_score, min(mv_num * (step + 1), len(u_data)))  # 从所有 u_data 里面选
        l_data, select_pre = tagper.move_unlabel_to_label_cpu(selected_idx, pred_y, u_data)
        tapger_end = time.time()

        tagper_file.write(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} len(l_data):{}  label_pre:{:.2%} select_pre:{:.2%}\n".format(
                int(step + 1), mAP, top1, top5, top10, top20, len(l_data), label_pre, select_pre))
        print(
            "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} len(l_data):{} label_pre:{:.2%} select_pre:{:.2%}\n".format(
                int(step + 1), mAP, top1, top5, top10, top20, len(l_data), label_pre, select_pre))

        if args.clock:
            reid_time = reid_end - reid_start
            tagper_time = tapger_end - tagper_start
            step_time = tapger_end + reid_start
            time_file.write(
                "step:{}  reid_time:{} tagper_time:{} step_time:{}\n".format(int(step + 1), reid_time, tagper_time,
                                                                             step_time))
            h, m, s = changetoHSM(step_time)
            print("this step is over, cost %02d:%02d:%02.6f" % (h, m, s))

    data_file.close()
    tagper_file.close()
    if (args.clock):
        exp_end = time.time()
        exp_time = exp_end - exp_start
        h, m, s = changetoHSM(exp_time)
        print("experiment is over, cost %02d:%02d:%02.6f" % (h, m, s))
        time_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snatch Strategy')
    parser.add_argument('-d', '--dataset', type=str, default='DukeMTMC-VideoReID', choices=datasets.names())  # s
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--step_size', type=int, default=55)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'data'))  # 加载数据集的根目录
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs'))  # 保持日志根目录
    parser.add_argument('--exp_name', type=str, default="gradully_supplement")
    parser.add_argument('--exp_order', type=str, default="1")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"],
                        default="Dissimilarity")  # 这个考虑要不要取消掉
    parser.add_argument('--max_frames', type=int, default=400)
    parser.add_argument('--clock', type=bool, default=True)  # 是否记时
    parser.add_argument('--gdraw', type=bool, default=False)  # 是否实时绘图
    # the key parameters is following
    parser.add_argument('--total_step', type=int, default=5)  # 默认总的五次迭代.
    parser.add_argument('--train_tagper_step', type=float, default=5)  # 用于训练 tagper的 step 数
    parser.add_argument('--is_baseline', type=bool, default=False)  # 默认不是baseline

    # 下面是暂时不知道用来做什么的参数
    parser.add_argument('-a', '--arch', type=str, default='avg_pool', choices=models.names())  # eug model_name
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--continuous', action="store_true")
    main(parser.parse_args())
