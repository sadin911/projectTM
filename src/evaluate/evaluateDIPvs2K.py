# -*- coding: utf-8 -*-
from method.AlgorithmRestnet import Restnet
import os
import glob2
import pickle
import numpy as np
class RunEvaluate:
    def __init__(self,algo):
        print("start init")
        self.algo=algo
        self.path_background_db=r"D:\project\trademark\train_and_test\train_and_test\test\**\\"
        self.path_probe_db=r"D:\project\trademark\projectTM\src\imageGroupALL"
        self.output_path = r'restnet/'
        self.iden_result=[]
        self.cmc_score=[]
        # self.bg_list=[]
        # for typee in ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif','*.jpeg'): 
        #     self.bg_list.extend(glob2.glob(self.path_background_db+typee))

        self.probe_list=os.listdir(self.path_probe_db)
        self.probe_image_list=[]
        self.probe_feature=[]
        
        for folder_name in self.probe_list:
            self.probe_image_list.append(glob2.glob(os.path.join(self.path_probe_db,folder_name,'*.jpg')))
        
        print("done init")
    def runEvaluate(self):
        print("evaluate")
        print("add bg image to db")
        # for i in range(len(self.bg_list)):
        #     feature = self.algo.feature_extract(self.bg_list[i])
        #     self.algo.enroll_to_DB(feature, "background_db_"+str(i))
        #     print(i)
        # self.algo.enroll()
        # self.algo.database = pickle.load(open(r'dbfeature.pk', 'rb'))
        # self.algo.ID = pickle.load(open(r'db_id.pk', 'rb')) 
        
        # for i in range(len(self.probe_image_list)):
        #     print(f'{i}')
        #     for j in range(len(self.probe_image_list[i])):
        #         feature = self.algo.feature_extract(self.probe_image_list[i][j])
        #         if(j==0):
        #             self.probe_feature.append(feature)
        #         else:
        #             self.algo.enroll_to_DB(feature, "probe_"+str(i)+"_"+str(j))
            
        #     print(i)
        self.algo.database = pickle.load(open(r'dbfeature_withprobe.pk', 'rb'))
        self.algo.ID = pickle.load(open(r'db_id_with_probe.pk', 'rb'))      
        self.probe_feature = pickle.load(open(r'db_probe_feature.pk', 'rb')) 
        
        self.algo.enroll()
        
        self.iden_result=[]
        for i in range(len(self.probe_image_list)):
            print(f'{i}')
            (iddd,score) = self.algo.retriev(self.probe_feature[i], 10000)
            for j in range(1,len(self.probe_image_list[i])):
                rank=iddd.index("probe_"+str(i)+'_'+str(j))+1
                self.iden_result.append(rank)
                print(rank)
        

        
        print("done enroll and ready to iden")
        
        
    def create_cmc_graph(self):
        print("compute cmc graph")
        self.cmc_score.append(self.iden_result.count(1))
        for rank_index in range(1,100):
            self.cmc_score.append(self.iden_result.count(rank_index+1)+self.cmc_score[-1])
        # 582 รูป
        cmc_result=np.array(self.cmc_score)/582*100
        # summmm=0
        # for i in range(len(self.probe_image_list)):
        #     summmm = summmm + len(self.probe_image_list[i])-1
        
if __name__ == '__main__':
    cat = RunEvaluate(Restnet())
    cat.runEvaluate()
    # with open("dbfeature.pk", 'wb') as file:
    #     pickle.dump(cat.algo.database, file)
    # with open("db_id_with_probe.pk", 'wb') as file:
    #     pickle.dump(cat.algo.ID, file)
    
    cat.create_cmc_graph()
    # with open("dbfeature_withprobe.pk", 'wb') as file:
    #     pickle.dump(cat.algo.database, file)
    # with open("db_probe_feature.pk", 'wb') as file:
    #     pickle.dump(cat.probe_feature, file)