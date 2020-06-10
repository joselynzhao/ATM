#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@AUTHOR:Joselyn Zhao
@CONTACT:zhaojing17@foxmail.com
@HOME_PAGE:joselynzhao.top
@SOFTWERE:PyCharm
@FILE:draw01.py
@TIME:2020/4/18 10:49
@DES:
'''
import argparse
import os.path as osp
import os
import codecs
# import mydatabase
import math
import matplotlib.pyplot as plt
import numpy as np
import json


def gain_data(file_name,exp_path,exp_name,group):
    f = codecs.open(file_name, 'r', 'utf-8')
    save_path = osp.join(exp_path,group,'format_data.txt')
    format_file = codecs.open(save_path,'w')
    datas = f.readlines()
    step = []
    mAP= []
    top1 = []
    top5 = []
    top10 = []
    top20 = []
    nums_selected = []
    select_percent = []
    label_pre = []
    select_pre = []

    for line in datas:
        line = line.strip()
        line = line.split(" ")
        for ll in line:
            data = ll.split(":")
            if(data[0]=="step"):
                step.append(int(data[1]))
            if(data[0]=="top1"):
                top1.append(round(float(data[1][:-1]),2))
            if (data[0] == "top5"):
                top5.append(round(float(data[1][:-1]), 2))
            if (data[0] == "top10"):
                top10.append(round(float(data[1][:-1]), 2))
            if (data[0] == "top20"):
                top20.append(round(float(data[1][:-1]), 2))
            if(data[0]=="nums_selected"):
                nums_selected.append(int(data[1]))
            if(data[0]=="selected_percent"):
                select_percent.append(round(float(data[1][:-1]),2))
            if(data[0] =="mAP"):
                mAP.append(round(float(data[1][:-1]),2))
            if(data[0]=="label_pre"):
                label_pre.append(round(float(data[1][:-1]),2))
            if(data[0] =="select_pre"):
                select_pre.append(round(float(data[1][:-1]),2))

    print("\"length\":{},".format(step[-1]))
    print("\"title\":\"{}\",".format(exp_name+'_'+group))
    print("\"step\":{},".format(step))
    print("\"top1\":{},".format(top1))
    print("\"top5\":{},".format(top5))
    print("\"top10\":{},".format(top10))
    print("\"top20\":{},".format(top20))
    print("\"mAP\":{},".format(mAP))
    print("\"nums_selected\":{},".format(nums_selected))
    print("\"select_percent\":{},".format(select_percent))
    print("\"label_pre\":{},".format(label_pre))
    print("\"select_pre\":{}\n".format(select_pre))
    format_file.write("\"length\":{},".format(step[-1]))
    format_file.write("\"title\":\"{}\",".format(exp_name+'_'+group))
    format_file.write("\"step\":{},".format(step))
    format_file.write("\"top1\":{},".format(top1))
    format_file.write("\"top5\":{},".format(top5))
    format_file.write("\"top10\":{},".format(top10))
    format_file.write("\"top20\":{},".format(top20))
    format_file.write("\"mAP\":{},".format(mAP))
    format_file.write("\"nums_selected\":{},".format(nums_selected))
    format_file.write("\"select_percent\":{},".format(select_percent))
    format_file.write("\"label_pre\":{},".format(label_pre))
    format_file.write("\"select_pre\":{}".format(select_pre))


def gain_data_f5(file_name,save_path,title ='none'):
    f = codecs.open(file_name, 'r', 'utf-8')
    format_file = codecs.open(save_path,'w')
    datas = f.readlines()
    step = []
    mAP= []
    top1 = []
    top5 = []
    top10 = []
    top20 = []
    label_pre = []
    select_pre = []

    for line in datas:
        line = line.strip()
        line = line.split(" ")
        for ll in line:
            data = ll.split(":")
            if(data[0]=="step"):
                step.append(int(data[1]))
            if(data[0]=="top1"):
                top1.append(round(float(data[1][:-1]),2))
            if (data[0] == "top5"):
                top5.append(round(float(data[1][:-1]), 2))
            if (data[0] == "top10"):
                top10.append(round(float(data[1][:-1]), 2))
            if (data[0] == "top20"):
                top20.append(round(float(data[1][:-1]), 2))
            if(data[0] =="mAP"):
                mAP.append(round(float(data[1][:-1]),2))
            if(data[0]=="label_pre"):
                label_pre.append(round(float(data[1][:-1]),2))
            if(data[0] =="select_pre"):
                select_pre.append(round(float(data[1][:-1]),2))

    print("\"length\":{},".format(step[-1]))
    print("\"title\":\"{}\",".format(title))
    print("\"step\":{},".format(step))
    print("\"top1\":{},".format(top1))
    print("\"top5\":{},".format(top5))
    print("\"top10\":{},".format(top10))
    print("\"top20\":{},".format(top20))
    print("\"mAP\":{},".format(mAP))
    print("\"label_pre\":{},".format(label_pre))
    print("\"select_pre\":{}\n".format(select_pre))
    format_file.write("\"length\":{},".format(step[-1]))
    format_file.write("\"title\":\"{}\",".format(title))
    format_file.write("\"step\":{},".format(step))
    format_file.write("\"top1\":{},".format(top1))
    format_file.write("\"top5\":{},".format(top5))
    format_file.write("\"top10\":{},".format(top10))
    format_file.write("\"top20\":{},".format(top20))
    format_file.write("\"mAP\":{},".format(mAP))
    format_file.write("\"label_pre\":{},".format(label_pre))
    format_file.write("\"select_pre\":{}".format(select_pre))

def summary_gradually_compare(compare_list,compare_item,save_path): #compare_item 是一个item 的list
    len_list = len(compare_item) # 有多少的item 就有多少张子图
    raw  = math.floor(pow(len_list,0.5))
    col = math.ceil(len_list/raw)
    print("col:{} , raw:{}".format(col, raw))
    unit_size = 8
    plt.figure(figsize=(4 * unit_size, 2 * unit_size), dpi=100)
    plt.subplots_adjust(hspace=0.3)  # 调整子图间距
    for i in range(len_list): #遍历每一个item
        plt.subplot(raw, col, i + 1)
        item  = compare_item[i]
        max_len = 0
        for train_name in compare_list:
            train_len = train_name["length"]
            if(max_len<train_len):
                max_len = train_len
            max_point = np.argmax(train_name[item])
            plt.annotate(str(train_name[item][max_point]), xy=(max_point + 1, train_name[item][max_point]))
            x = np.linspace(1, train_name["length"] , train_name["length"])
            plt.plot(x,train_name[item],label=train_name["title"],marker='o')
        # plt.xticks(range(1, max_len+1, round(max_len/10)))
        plt.xlabel("steps")
        plt.ylabel("value(%)")
        plt.title(item)
        if i == 1:
            # plt.legend(loc="best")
            plt.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=len(compare_list))  # 1
        # if compare_item in ["select_pre","train_pre"]:
        #     plt.legend(loc="upper right")
        # else: plt.legend(loc="lower right")
    plt.savefig(save_path,bbox_inches="tight")
    plt.show()


def draw_relative_curve(compare_item,compare_x_axis,compare_list,save_path,exp_name):
    len_list = len(compare_item)  # 有多少的item 就有多少张子图
    raw = math.floor(pow(len_list, 0.5))
    col = math.ceil(len_list / raw)
    print("col:{} , raw:{}".format(col, raw))
    unit_size = 8
    plt.figure(figsize=(4 * unit_size, 2 * unit_size), dpi=100)
    plt.subplots_adjust(hspace=0.3)  # 调整子图间距
    for i in range(len_list): #遍历每一个item
        plt.subplot(raw, col, i + 1)
        item  = compare_item[i]
        max_len = 0
        for train_name in compare_list:
            train_len = train_name["length"]
            xx = train_name["select_percent"]
            if(max_len<train_len):
                max_len = train_len
            max_point = np.argmax(train_name[item])
            if item in ["select_pre"]:
                max_point = np.argmin(train_name[item])
            if compare_x_axis == 'select_percent':
                plt.annotate(str(train_name[item][max_point]), xy=(xx[max_point], train_name[item][max_point]))
            else:plt.annotate(str(train_name[item][max_point]), xy=(max_point + 1, train_name[item][max_point]))
            x = train_name[compare_x_axis]
            plt.plot(x,train_name[item],label=train_name["title"],marker='o')
        # plt.xticks(range(1, max_len+1, round(max_len/10)))
        plt.xlabel(compare_x_axis)
        plt.ylabel("value(%)")
        plt.title(item)
        if i == 1:
            # plt.legend(loc="best")
            plt.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=len(compare_list))  # 1
        # if compare_item in ["select_pre","train_pre"]:
        #     plt.legend(loc="upper right")
        # else: plt.legend(loc="lower right")
    plt.savefig(osp.join(save_path,exp_name+'_'+compare_x_axis),bbox_inches="tight")
    plt.show()


def get_top_value(group,format_data,topvalue_file):
    print(format_data)
    top1s = format_data["top1"]
    top5s = format_data["top5"]
    top10s = format_data["top10"]
    top20s = format_data["top20"]
    mAPs = format_data["mAP"]
    label_pres = format_data["label_pre"]
    select_pres = format_data["select_pre"]
    # print(top1s)
    max_top1 = np.max(top1s)
    max_top5 = np.max(top5s)
    max_top10 = np.max(top10s)
    max_top20 = np.max(top20s)
    max_mAP = np.max(mAPs)
    mean_label_pre = round(np.mean(label_pres),2)
    mean_selecet_pre = round(np.mean(select_pres),2)
    topvalue_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(group,max_mAP,max_top1,max_top5,max_top10,max_top20,mean_label_pre,mean_selecet_pre))


def main(args):
    if args.function == 1:  # 提取文件内容,手动存储到data文件
        exp_path = osp.join('logs',args.dataset,args.exp_name)
        group_list = os.listdir(exp_path)
        group_list.sort()
        if args.exp_order:
            group_list = args.exp_order
        for group in group_list:
            if os.path.isfile(osp.join(exp_path, group)): continue  # 跳过图片文件
            data_file = osp.join(exp_path,group,'data.txt')
            group_files = os.listdir(osp.join(exp_path,group))
            if "format_data.txt" in group_files:
                print("format_data.txt in group {} is exsist!".format(group))
                continue
            gain_data(data_file,exp_path,args.exp_name,group)  # 数据文件, 实验名称,组号 生成格式化数据


    elif args.function == 2:  # 利用my date 里面的数据绘图
        file_path = osp.join("logs",args.dataset,args.exp_name) # 图像的存储路劲
        group_list = os.listdir(file_path)
        group_list.sort()
        if args.exp_order:
            group_list = args.exp_order
        compare_item = ["mAP", "top1", "top5", "top10", "top20", "label_pre"]
        compare_train = []
        for group in group_list:
            if os.path.isfile(osp.join(file_path,group)): continue  #跳过图片文件
            format_file = codecs.open(osp.join(file_path,group,'format_data.txt'),'r', 'utf-8')
            format_data = '{'+format_file.read()+'}' # 使其变为字典
            format_data = eval(format_data)  #转为真正的字典
            compare_train.append(format_data)
        summary_gradually_compare(compare_train,compare_item,file_path)

    elif args.function == 3:   #提取极值数据
        file_path = osp.join("logs",args.dataset,args.exp_name) # 文件的存储路劲
        topvalue_file = codecs.open(osp.join(file_path, 'topvalue.txt'), mode='a')  # 如果没有自行创建
        topvalue_file.write("order\tmAP\ttop1\ttop5\ttop10\ttop20\tmLEA\tmSA\n")
        group_list = os.listdir(file_path)
        group_list.sort()
        if args.exp_order:
            group_list = args.exp_order
            # print(group_list)
        for group in group_list:
            if  os.path.isfile(osp.join(file_path,group)): continue  #跳过图片文件
            format_file = codecs.open(osp.join(file_path, group, 'data_form.txt'), 'r', 'utf-8')
            format_data = '{' + format_file.read() + '}'  # 使其变为字典
            format_data = eval(format_data)  # 转为真正的字典
            # print(format_data)
            get_top_value(group,format_data,topvalue_file)
        topvalue_file.close()

    elif args.function ==4:  # 分别以标签准确率和选择准确率为横坐标,绘制 mAP和top1等性能的曲线图.
        file_path = osp.join("logs", args.dataset, args.exp_name)  # 图像的存储路劲
        group_list = os.listdir(file_path)
        group_list.sort()
        if args.exp_order:
            group_list = args.exp_order
        compare_train = []
        for group in group_list:
            if os.path.isfile(osp.join(file_path, group)): continue  # 跳过图片文件
            format_file = codecs.open(osp.join(file_path, group, 'format_data.txt'), 'r', 'utf-8')
            format_data = '{' + format_file.read() + '}'  # 使其变为字典
            format_data = eval(format_data)  # 转为真正的字典
            compare_train.append(format_data)
        compare_item = []
        compare_item.append(["mAP", "top1", "top5", "top10", "top20", "label_pre", "select_pre","nums_selected"])
        # compare_item = [["mAP", "top1", "top5", "top10", "top20", "select_pre", "select_percent"],
        #                 ["mAP", "top1", "top5", "top10", "top20", "label_pre", "select_percent"],
        #                 ["mAP", "top1", "top5", "top10", "top20", "label_pre", "select_pre"]]
        compare_x_axis = ["select_percent"]
        for i in range(len(compare_x_axis)):
            draw_relative_curve(compare_item[i],compare_x_axis[i],compare_train,file_path,args.exp_name)



    elif args.function == 5: # generate formate file for reid and tagper in the special exp_order
        file_path = osp.join("logs",args.dataset, args.exp_name,args.exp_order[0])
        data_file = osp.join(file_path,'data.txt')
        tagper_file = osp.join(file_path,'tagper1_data.txt')
        gain_data_f5(data_file,osp.join(file_path,"data_form.txt"),title=args.exp_name+'_'+args.exp_order[0]+'_reid')
        # gain_data_f5(tagper_file,osp.join(file_path,"tagper_form.txt"),title=args.exp_name+'_'+args.exp_order[0]+'_tagper')

    elif args.function ==6:  # draw for the special exp_order reid vs tagper
        file_path = osp.join("logs", args.dataset, args.exp_name, args.exp_order[0])
        reid_file = codecs.open(osp.join(file_path,'data_form.txt'),'r', 'utf-8')
        tagper_file = codecs.open(osp.join(file_path,'tagper_form.txt'),'r', 'utf-8')
        reid_data = eval('{'+reid_file.read()+'}')
        tagper_data = eval('{'+tagper_file.read()+'}')
        compare_list =[reid_data,tagper_data]
        compare_item = ["mAP", "top1", "top5", "top10", "top20", "label_pre",'select_pre']
        summary_gradually_compare(compare_list, compare_item, osp.join(file_path,'reidvstagper'))

    elif args.function == 7 : # 多atm 对比 reid
        file_path = osp.join("logs", args.dataset, args.exp_name)  # 图像的存储路劲
        group_list = os.listdir(file_path)
        group_list.sort()
        if args.exp_order:
            group_list = args.exp_order
        compare_item = ["mAP", "top1", "top5", "top10", "top20", "label_pre",'select_pre']
        compare_train = []
        formfile_name = 'data_form.txt'
        if args.reidortagper == 'tagper':
            formfile_name = 'tagper_form.txt'
        for group in group_list:
            if os.path.isfile(osp.join(file_path, group)): continue  # 跳过图片文件
            format_file = codecs.open(osp.join(file_path, group, formfile_name), 'r', 'utf-8')
            format_data = eval('{' + format_file.read() + '}')  # 使其变为字典
            compare_train.append(format_data)
        #尝试和非atm方法对比.
        none_atm = codecs.open(osp.join('logs',args.dataset,'gradually_11step','0','format_data.txt'),'r','utf-8')
        # compare_train.append(eval('{'+none_atm.read()+'}'))
        summary_gradually_compare(compare_train, compare_item, file_path+args.reidortagper)

    # elif args.function == 8 :# 与非atm方法对比.





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snatch Strategy')
    parser.add_argument('--dataset', type=str, default='DukeMTMC-VideoReID')
    parser.add_argument('--exp_name', type=str, default='snatch')
    parser.add_argument('--exp_order',type=list,default=None)
    parser.add_argument('--function', type=int, default=2)
    parser.add_argument('--reidortagper',type=str,default='reid')
    main(parser.parse_args())



