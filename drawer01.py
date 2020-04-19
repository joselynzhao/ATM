import os


class drawer01():
    def __inti__(self,dataset = 'DukeMTMC-VideoReID', exp_name = 'ATM'):
        self.dataset =dataset
        self.exp_name = exp_name
        self.exp_path = osp.join('log',self.dataset,self.exp_name)
        self.__get_group_list()
    def __get_group_list(self):
        all_content = os.listdir(self.exp_path).sort()
        self.group_list =[group for group in all_content if not os.path.isfile(osp.join(self.exp_path,group))]
    def generate_formdata(self,exp_order): # 指定数据组生成格式数据
        pass
    def generate_formdata_forall(self):
        for group in self.group_list:
            group_files = os.listdir(osp.join(self.exp_path,group))
            data_file = osp.join(self.exp_path, group, 'data.txt')


