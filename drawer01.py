import os
import codecs
import os.path as osp
import math
import matplotlib.pyplot as plt
import numpy as np



class drawer01():
    def __init__(self,dataset = 'DukeMTMC-VideoReID', exp_name = 'atm'):
        self.dataset =dataset
        self.exp_name = exp_name
        self.exp_path = osp.join('logs',self.dataset,self.exp_name)
        # self.group_list = group_list
        # self.__get_group_list()
    # def __get_group_list(self):
    #     all_content = os.listdir(self.exp_path)
    #     self.group_list =[group for group in all_content if not os.path.isfile(osp.join(self.exp_path,group))]
    def generate_formdata_for_group_list(self,group_list):
        self.group_list = group_list
        for group in self.group_list:
            group = str(group)
            data_file = osp.join(self.exp_path, group, 'data.txt')
            self.__for_generate_formdata(data_file,group,'format_data.txt')
            tagper_file = osp.join(self.exp_path, group, 'tagper_data.txt')
            self.__for_generate_formdata(tagper_file, group, 'format_tagper.txt')

    def  __for_generate_formdata(self,data_file,group,out_name):
        file = codecs.open(data_file,'r','utf-8')
        save_path = osp.join(self.exp_path,group,out_name)
        format_file = codecs.open(save_path,'w')
        datas = file.readlines()
        for i in range(len(datas)):
            datas[i] = datas[i].strip().split(' ')
            datas[i] =[k.split(':')[-1].strip('%') for k in datas[i]]  #只取数值
        # datas = np.aray(datas)
        print(datas)
        name_list = ['step','mAP','Rank-1','Rank-5','Rank-10','Rank_20','num_selected','label_pre','select_pre']
        format_file.write("\"length\":{},".format(len(datas)))
        format_file.write("\"title\":\"{}\"".format(self.exp_name + '_' + group))
        for i in range(len(name_list)):
            data = [float(datas[k][i]) for k in range(len(datas))]
            format_file.write(",\"{}\":{}".format(name_list[i],data))

if __name__ =='__main__':
    drawer = drawer01()
    # drawer.init()
    drawer.generate_formdata_for_group_list([0,1,2,3])



