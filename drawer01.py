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

    def generate_formdata_for_group_list(self,group_list):
        for group in group_list:
            group = str(group)
            data_file = osp.join(self.exp_path, group, 'data.txt')
            self.__for_generate_formdata(data_file,group,'format_data.txt')
            tagper_file = osp.join(self.exp_path, group, 'tagper_data.txt')
            self.__for_generate_formdata(tagper_file, group, 'format_tagper.txt')
    def compare_reid_and_tagper(self,group_list,compare_item=['mAP','Rank-1','Rank-5','Rank-10','Rank-20','num_selected','label_pre','select_pre'],unit_size =4,dpi=100,hspace=0.3):
        for group in group_list:
            try:
                reid_file = codecs.open(osp.join(self.exp_path,str(group),'format_data.txt'),'r','utf-8')
                tagper_file = codecs.open(osp.join(self.exp_path,str(group),'format_tagper.txt'),'r','utf-8')
            except FileNotFoundError:
                self.generate_formdata_for_group_list([group])
                reid_file = codecs.open(osp.join(self.exp_path, str(group), 'format_data.txt'), 'r', 'utf-8')
                tagper_file = codecs.open(osp.join(self.exp_path, str(group), 'format_tagper.txt'), 'r', 'utf-8')
            reid_data = eval('{' + reid_file.read() + '}')
            tagper_data = eval('{' + tagper_file.read() + '}')
            compare_list = [reid_data, tagper_data]
            out_name ='reidvstagper_{}'.format(group)
            self.__draw_compre_for_list(compare_list,compare_item,out_name,unit_size,dpi,hspace,is_reidvstagper=1)

    def compare_train_list(self,train_list,is_tagper=0,compare_item=['mAP','Rank-1','Rank-5','Rank-10','Rank-20','num_selected','label_pre','select_pre'],unit_size =4,dpi=100,hspace=0.3):
        file_name = 'format_tagper.txt' if is_tagper else 'format_data.txt'
        compare_list = []
        for train in train_list:
            try:
                file_info = codecs.open(osp.join(self.exp_path,str(train),file_name),'r','utf-8')
            except FileNotFoundError:
                self.generate_formdata_for_group_list([train])
                file_info = codecs.open(osp.join(self.exp_path, str(train), file_name), 'r', 'utf-8')
            file_data = eval('{' + file_info.read() + '}')
            compare_list.append(file_data)
        baseline10 = codecs.open(osp.join('logs',self.dataset,'baseline','EF10','format_data.txt'),'r','utf-8')
        baseline15 = codecs.open(osp.join('logs',self.dataset,'baseline','EF15','format_data.txt'),'r','utf-8')
        baseline10 = eval('{' + baseline10.read() + '}')
        baseline15 = eval('{' + baseline15.read() + '}')
        compare_list.extend([baseline10,baseline15])
        out_name = 'comparetrains_tagper_{}'.format(train_list) if is_tagper else 'comparetrains_reid_{}'.format(train_list)
        self.__draw_compre_for_list(compare_list,compare_item,out_name,unit_size,dpi,hspace)

    def get_top_value_for_all(self,is_tagper=0):  #自动捕捉所有的训练
        dictionary = os.listdir(self.exp_path)
        group_list = [one for one in dictionary if not os.path.isfile(osp.join(self.exp_path,one))]
        group_list.sort()
        file_name = 'format_data.txt' if not is_tagper else 'format_tagper.txt'
        out_name = 'reid_topvalue.txt' if not is_tagper else 'tagper_topvalue.txt'
        items = ['step','mAP','Rank-1','Rank-5','Rank-10','Rank-20','label_pre','select_pre']
        out_file = codecs.open(osp.join(self.exp_path,out_name), 'w')
        out_file.write('group')
        for item in items:
            out_file.write('\t{}'.format(item))
        out_file.write('\n')
        for group in group_list:
            try:
                file_info = codecs.open(osp.join(self.exp_path,str(group),file_name),'r','utf-8')
            except FileNotFoundError:
                self.generate_formdata_for_group_list([group])
                file_info = codecs.open(osp.join(self.exp_path, str(group), file_name), 'r', 'utf-8')
            file_data = eval('{' + file_info.read() + '}')
            max_data =[max(file_data[item]) for item in items]
            out_file.write(group)
            for data in max_data:
                out_file.write('\t{}'.format(data))
            out_file.write('\n')
        out_file.close()

    def __draw_compre_for_list(self,compare_list,compare_item,out_name,unit_size,dpi,hspace,is_reidvstagper=0):
        item_num = len(compare_item)
        raw = math.floor(pow(item_num,0.5)) #为了让图尽可能方正
        col = math.ceil(item_num/raw)
        plt.figure(figsize=(4*unit_size,2*unit_size),dpi=dpi)
        plt.subplots_adjust(hspace=hspace)
        for i in range(item_num):
            plt.subplot(raw,col,i+1)
            item = compare_item[i]
            max_len = max([train['length'] for train in compare_list]) #    求这个max_len来做什么呢,好像没有用呀
            for train in compare_list:
                max_point = np.argmax(train[item])
                plt.annotate(str(train[item][max_point]),xy=(max_point+1,train[item][max_point]))
                x = np.linspace(1,train['length'],train['length'])
                plt.plot(x,train[item],label=train['title'],marker='o')
            plt.xlabel('steps')
            plt.ylabel('value(%)')
            plt.title(item)
            if i==1:
                if is_reidvstagper:
                    plt.legend(['reid','tagper'],loc='center', bbox_to_anchor=(0.5, 1.2), ncol=len(compare_list))  # 1
                else:
                    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=len(compare_list))  # 1
        plt.savefig(osp.join(self.exp_path,out_name),bbox_inches='tight')
        plt.show()


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
        name_list = ['step','mAP','Rank-1','Rank-5','Rank-10','Rank-20','num_selected','label_pre','select_pre']
        format_file.write("\"length\":{},".format(len(datas)))
        format_file.write("\"title\":\"{}\"".format(self.exp_name + '_' + group))
        for i in range(len(name_list)):
            data = [float(datas[k][i]) for k in range(len(datas))]
            format_file.write(",\"{}\":{}".format(name_list[i],data))

if __name__ =='__main__':
    drawer = drawer01()
    # drawer.init()
    # drawer.generate_formdata_for_group_list([0,1,2,3])
    # drawer.compare_reid_and_tagper([4,5])
    drawer.compare_train_list([1,4])
    # drawer.get_top_value_for_all(is_tagper=0)

