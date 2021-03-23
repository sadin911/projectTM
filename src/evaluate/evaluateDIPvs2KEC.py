# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:05:10 2021

@author: chonlatid.d
"""

from method.AlgorithmInceptionV3ECDIP import InceptionV3ENDIP, progressBar
import os
import glob2
import pickle
import numpy as np
from CMC import CMC
import pandas as pd

class RunEvaluate:
    def __init__(self,algo):
        print("start init")
        self.algo=algo
        self.bg_list= pd.read_csv(r'D:/datasets/LSLOGO/List/test_images_root.txt', delimiter = "\t",header=None)[0].tolist()
        self.bg_list = [r'D:/datasets/LSLOGO/Logo-2K+/' + x for x in  self.bg_list]
        self.path_probe_db=r"D:\datasets\TradeMark\trainingSet\imageGroupReName"
        self.output_path = r'restnet/'
        self.iden_result=[]
        self.cmc_score=[]
     
        self.probe_list=os.listdir(self.path_probe_db)
        self.probe_image_list=[]
        self.probe_feature=[]
        
        for folder_name in self.probe_list:
            self.probe_image_list.append(glob2.glob(os.path.join(self.path_probe_db,folder_name,'*.jpg')))
        
        print("done init")
     
    def runEvaluate(self):
        print("evaluate")
        print("add bg image to db")
        
        temp_probelist = []
        for i in range(len(self.probe_image_list)):
           for j in range(len(self.probe_image_list[i])):
               temp_probelist.append(self.probe_image_list[i][j])
               
        feature = self.algo.feature_extract_batch(temp_probelist, 100, 'DIPECProbe.pkl')
        rindex = 0
        for i in range(len(self.probe_image_list)):
            for j in range(len(self.probe_image_list[i])):
                if(j==0):
                    self.probe_feature.append(feature[rindex])
                else:
                    self.algo.enroll_to_DB(feature[rindex], "probe_"+str(i)+"_"+str(j))
                rindex+=1
            print('', end='\r')
            progressBar(i+1,len(self.probe_image_list),20)
        self.algo.enroll()
        
     
        feature_list = self.algo.feature_extract_batch(self.bg_list[:], 1000,'DIPECBackground.pkl')  
        for i in range(feature_list.shape[0]):
            self.algo.enroll_to_DB(feature_list[i,:], "background_db_"+str(i))
            
        print("done add bg to db")
        self.algo.enroll()
        
       
        
        
        # self.algo.database = pickle.load(open(r'dbfeature_withprobe.pk', 'rb'))
        # self.algo.ID = pickle.load(open(r'db_id_with_probe.pk', 'rb'))      
        # self.probe_feature = pickle.load(open(r'db_probe_feature.pk', 'rb')) 
        
        self.algo.clear_model()
        self.iden_result=[]
        print('retrival step')
        for i in range(len(self.probe_feature)):
            print('', end='\r')
            progressBar(i+1,len(self.probe_feature),20)
            (iddd,score) = self.algo.retriev(self.probe_feature[i], 100)
            for j in range(1,len(self.probe_image_list[i])):
                rank=iddd.index("probe_"+str(i)+'_'+str(j))+1
                self.iden_result.append(rank)
        print("done enroll and ready to iden")
        
        
    def create_cmc_graph(self):
        print("compute cmc graph")
        self.cmc_score.append(self.iden_result.count(1))
        for rank_index in range(1,100):
            self.cmc_score.append(self.iden_result.count(rank_index+1)+self.cmc_score[-1])
        # 582 รูป
        cmc_result=np.array(self.cmc_score)/len(self.probe_image_list)
        return cmc_result
        # summmm=0
        # for i in range(len(self.probe_image_list)):
        #     summmm = summmm + len(self.probe_image_list[i])-1
        
if __name__ == '__main__':
    cat = RunEvaluate(InceptionV3ENDIP())
    cat.runEvaluate()
    # with open("dbfeature.pk", 'wb') as file:
    #     pickle.dump(cat.algo.database, file)
    # with open("db_id_with_probe.pk", 'wb') as file:
    #     pickle.dump(cat.algo.ID, file)
    cmc_data = cat.create_cmc_graph()
    
    cmc_dict = {
    'ECDIP': cmc_data
}
    
    
    cmc = CMC(cmc_dict)
    cmc.plot(title = 'CMC on Search Rank\n', rank=100,
             xlabel='Rank',
             ylabel='Hit Rate', show_grid=False)
    # with open("dbfeature_withprobe.pk", 'wb') as file:
    #     pickle.dump(cat.algo.database, file)
    # with open("db_probe_feature.pk", 'wb') as file:
    #     pickle.dump(cat.probe_feature, file)