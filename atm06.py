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


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    one_shot, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(
        dataset_all.name))

    def sampleing_number_curve(step):  # p = 1时就是线性曲线
        yr = min(math.floor(pow(step / args.total_step, args.p)), len(u_data))
        yt = 2 * yr if 2 * yr < len(u_data) else 0
        return yr, yt  # 当yt=0的时候,就说明 训练进入了单循环模式

    def train_epoch(yr): #只有训练reid的时候采用
        ep_k = math.ceil((len(one_shot)+yr)/len(one_shot))
        times = max(math.floor(args.epoch/ep_k),1)
        return ep_k,times

    Ep = [] # 经验
    AE = [] # 辅助经验
    PE = [] # 实践经验


    # 省略掉oneshot训练部分

    # 输出实验信息
    print("{}/{} is training with {}, the max_frames is {}, and will be saved to {}".format(args.exp_name,args.exp_order,args.dataset,args.max_frames,args.logs_dir))
    # 输出超参信息
    print("parameters are setted as follows")
    print("\ttotal_step:\t{}".format(args.total_step))
    print("\tepoch:\t{}".format(args.epoch))
    print("\tstep_size:\t{}".format(args.step_size))
    print("\tbatch_size:\t{}".format(args.batch_size))

    # 指定输出文件
    # 第三部分要说明关键参数的设定
    reid_path = osp.join(args.logs_dir, args.dataset, args.exp_name, args.exp_order)
    sys.stdout = Logger(osp.join(reid_path, 'log' + time.strftime(".%m_%d_%H-%M-%S") + '.txt'))
    data_file = codecs.open(osp.join(reid_path, 'data.txt'), mode='a')
    time_file = codecs.open(osp.join(reid_path, 'time.txt'), mode='a')
    tagper_file = codecs.open(osp.join(reid_path, "tagper_data.txt"), mode='a')

    # initial the EUG algorithm

    # 注意不要去破坏公共部分的代码
    reid = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode, num_classes=dataset_all.num_train_ids,
               data_dir=dataset_all.images_dir, l_data=one_shot, u_data=u_data, save_path=reid_path,
               max_frames=args.max_frames)
    reid.resume(osp.join(reid_path, 'Dissimilarity_step_0.ckpt'), 0)

    # 初始化循环模式
    iter_mode = 2 # 2双循环模式 1单循环模式

    # 开始循环
    for step in range(1,args.total_step+1):
        # 获取采样数量
        num_reid,num_tagper = sampleing_number_curve(step)
        train_ep,train_times = train_epoch(num_reid)
        # 克隆种子得到标签器
        iter_mode = 2 if num_tagper else 1

        print("### step {} is training: num_reid={},num_tagper={}, train_ep={},train_times={}".format(step,num_reid,num_tagper,train_ep,train_times))
        if iter_mode == 2:
            tagper = EUG(model_name=args.arch, batch_size=args.batch_size, mode=args.mode,
                         num_classes=dataset_all.num_train_ids,
                         data_dir=dataset_all.images_dir, l_data=one_shot, u_data=u_data, save_path=reid_path,
                         max_frames=args.max_frames)
            tagper.resume(osp.join(reid_path, 'Dissimilarity_step_{}.ckpt'.format(step-1)), step-1)

            # 实践
            PE_pred_y, PE_pred_score, PE_label_pre = reid.estimate_label_atm3(u_data, Ep, one_shot)  # 针对u_data进行标签估计
            selected_idx_RR = reid.select_top_data(PE_pred_score, num_reid)
            select_pre_R = reid.get_select_pre(selected_idx_RR, PE_pred_y, u_data)
            selected_idx_RT = reid.select_top_data(PE_pred_score, num_tagper)
            select_pre_T = reid.get_select_pre(selected_idx_RT, PE_pred_y, u_data)


            # 训练tagper
            new_train_data = tagper.generate_new_train_data_only(selected_idx_RT, PE_pred_y,
                                                                 u_data)  # 这个选择准确率应该是和前面的label_pre是一样的.
            train_tagper_data = one_shot + new_train_data
            tagper.train(train_tagper_data, step, tagper=1, epochs=args.epoch, step_size=args.step_size, init_lr=0.1)

            # 性能评估
            mAP, top1, top5, top10, top20 = reid.evaluate(dataset_all.query, dataset_all.gallery)

            AE_pred_y, AE_pred_score, AE_label_pre = tagper.estimate_label_atm3(u_data, Ep, one_shot)  # 针对u_data进行标签估计
            tagper_file.write(
                "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} num_tagper:{} label_pre:{:.2%}\n".format(
                    int(step), mAP, top1, top5, top10, top20,num_tagper,AE_label_pre))
            print(
                "step:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%} num_tagper:{} label_pre:{:.2%}\n".format(
                    int(step), mAP, top1, top5, top10, top20, num_tagper, AE_label_pre))

            #下面需要进行知识融合 KF
            AEs = normalization(AE_pred_score)
            PEs = normalization(PE_pred_score)
            KF =[PE_pred_y[i]==AE_pred_y[i] for i in range(len(u_data))]
            KF_score= [KF[i]*(PEs[i]+AEs[i])+(1-KF[i])*abs(PEs[i]-AEs[i]) for i in range(len(u_data))]
            KF_label = [KF[i]*PE_pred_y[i]+(1-KF[i])*(PE_pred_y[i] if PEs[i]>=AEs[i] else AE_pred_y[i]) for i in range(len(u_data))]
            u_label = np.array([label for _, label, _, _ in u_data])
            is_label_right = [1 if u_label[i]==KF_label[i] else 0 for i in range(len(u_label))]
            KF_label_pre = sum(is_label_right)/len(u_label)

            #获取Ep
            selected_idx_Ep = tagper.select_top_data(KF_score,num_reid)
            Ep,Ep_select_pre = tagper.move_unlabel_to_label_cpu(selected_idx_Ep,KF_label,u_data)
            data_file.write("step:{} num_reid:{} select_pre_R:{:.2%} select_pre_T:{:.2%} KF_label_pre:{:.2%} Ep_select_pre:{:.2%}\n".format(step,num_reid,select_pre_R,select_pre_T,KF_label_pre,Ep_select_pre))
            print("step:{} num_reid:{} select_pre_R:{:.2%} select_pre_T:{:.2%} KF_label_pre:{:.2%} Ep_select_pre:{:.2%}\n".format(step,num_reid,select_pre_R,select_pre_T,KF_label_pre,Ep_select_pre))


        elif iter_mode==1:
            PE_pred_y, PE_pred_score, PE_label_pre = reid.estimate_label_atm3(u_data, Ep, one_shot)  # 针对u_data进行标签估计
            selected_idx_RR = reid.select_top_data(PE_pred_score, num_reid)
            Ep, Ep_select_pre = reid.move_unlabel_to_label_cpu(selected_idx_RR, PE_pred_y, u_data)
            data_file.write(
                "step:{} num_reid:{} Ep_select_pre:{:.2%}\n".format(
                    step, num_reid, Ep_select_pre)) # Ep_select_pre 和select_pre_R 是一样的.
            print(
                "step:{} num_reid:{} Ep_select_pre:{:.2%}\n".format(
                    step, num_reid, Ep_select_pre))  # Ep_select_pre 和select_pre_R 是一样的.

        # 训练种子
        train_seed_data = Ep + one_shot
        for i in range(train_times):
            reid.train_atm06(train_seed_data, step, i, epochs=train_ep, step_size=args.step_size, init_lr=0.1)
            mAP, top1, top5, top10, top20 = reid.evaluate(dataset_all.query, dataset_all.gallery)
            data_file.write(
                "step:{} times:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%}\n".format(
                    int(step + 1), i, mAP, top1, top5, top10, top20))
            print(
                "step:{} times:{} mAP:{:.2%} top1:{:.2%} top5:{:.2%} top10:{:.2%} top20:{:.2%}\n".format(
                    int(step + 1), i, mAP, top1, top5, top10, top20))



        if args.clock:
            reid_time = reid_end - reid_start
            tagper_time = tapger_end - tagper_start
            step_time = tapger_end - reid_start
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
    parser = argparse.ArgumentParser(description='ATM')
    parser.add_argument('-d', '--dataset', type=str, default='DukeMTMC-VideoReID', choices=datasets.names())  # s

    working_dir = os.path.dirname(os.path.abspath(__file__))  # 有用绝对地址
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'data'))  # 加载数据集的根目录
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'logs'))  # 保持日志根目录
    parser.add_argument('--exp_name', type=str, default="atm")
    parser.add_argument('--exp_order', type=str, default="0")
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--mode', type=str, choices=["Classification", "Dissimilarity"],
                        default="Dissimilarity")  # 这个考虑要不要取消掉
    parser.add_argument('--max_frames', type=int, default=400)
    parser.add_argument('--clock', type=bool, default=True)  # 是否记时
    parser.add_argument('--is_baseline', type=bool, default=False)  # 默认不是baseline
    # the key parameters is following
    parser.add_argument('--total_step', type=int, default=5)  # 默认总的五次迭代.
    parser.add_argument('--train_tagper_step', type=float, default=3)  # 用于训练 tagper的 step 数
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--step_size', type=int, default=55)
    parser.add_argument('-b', '--batch_size', type=int, default=16)


    '''new'''
    parser.add_argument('--p', type=int, default=1)  # 采样曲线的指数

    # 下面是暂时不知道用来做什么的参数
    parser.add_argument('-a', '--arch', type=str, default='avg_pool', choices=models.names())  # eug model_name
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--l', type=float)
    parser.add_argument('--continuous', action="store_true")
    main(parser.parse_args())


    '''
    python3.6 atm03.py  --total_step 5 --train_tagper_step 3 --exp_order 2
    python3.6 atm03.py  --total_step 8 --train_tagper_step 4 --exp_order 3
    '''