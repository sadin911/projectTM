# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:07:09 2021

@author: chonlatid.d
"""

from method.AlgorithmInceptionV3ECDIP import InceptionV3ENDIP
import os
import glob2
import pickle
import numpy as np
import pandas as pd


class RunEvaluate:
    def __init__(self,algo):
        print("[EvaluateInit]")
        
        self.algo=algo
        self.path_background_db= pd.read_csv(r'D:/datasets/LSLOGO/List/test_images_root.txt', delimiter = "\t",header=None)[0].tolist()
        self.path_probe_searchdb = r"D:/datasets/TradeMark/trainingSet/ImagesReName/DIP/Search/**/"
        self.path_probe_refdb = r"D:/datasets/TradeMark/trainingSet/ImagesReName/DIP/Reference/**/"
        self.path_background_db = [r'D:/datasets/LSLOGO/Logo-2K+/' + x for x in  self.path_background_db]
        
        self.output_path = r'Result/InceptionV3ENDIP/'
        self.iden_result=[]
        self.cmc_score=[]
        self.scores = []
        # self.bg_list=[]
        # for typee in ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif','*.jpeg'): 
        #     self.bg_list.extend(glob2.glob(self.path_background_db+typee))
        types = ('*.bmp', '*.jpg' ,'.*gif' ,'*.png' , '*.tif','*.jpeg')
        
        self.probe_list_search = []
        self.probe_list_ref = []
        for files in types:
            self.probe_list_search.extend(glob2.glob(os.path.join(self.path_probe_searchdb, files)))
            self.probe_list_ref.extend(glob2.glob(os.path.join(self.path_probe_refdb, files)))
        self.path_background_db.extend(self.probe_list_ref)
        self.probe_feature=[]
        
        print("done init")
        
    def runEvaluate(self):
        print("[evaluating]")
        
        print("[featureExtraction] search")
        template_search = self.algo.feature_extract_batch(self.probe_list_ref,100,pkl_save_path='feature_searchInceptionV3ECDIP.pkl')
        
        print("[featureExtraction] background")
        template_ref = self.algo.feature_extract_batch(self.probe_list_ref,100,pkl_save_path='feature_bgInceptionV3ECDIP.pkl')
        
        print("done enroll and ready to iden")
        for i in range(len(template_search)):
            search_id = self.probe_list_search[i].split('_')[0].split('\\')[-1]
            self.algo.enroll_to_DB(template_search[i],search_id)
        
        for i in range(len(template_ref)):
            ref_id = self.path_background_db[i].split('_')[0].split('\\')[-1]
            self.algo.enroll_to_DB(template_ref[i],ref_id)
            
        # self.scores = self.algo.match_1to1_batch(template_search, template_ref)
        for i in range(len(template_search)):
            (ids,score) = self.algo.retrieve(template_search[i], 100)
            for j in range(1,len(self.probe_list_search[i])):
                rank=ids.index("probe_"+str(i)+'_'+str(j))+1
                self.iden_result.append(rank)
                print(rank)
        
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
    cat = RunEvaluate(InceptionV3ENDIP())
    path_background_db = cat.path_background_db
    probe_list_search = cat.probe_list_search
    probe_list_ref = cat.probe_list_ref
    cat.runEvaluate()
    scores = cat.scores
    # with open("dbfeature.pk", 'wb') as file:
    #     pickle.dump(cat.algo.database, file)
    # with open("db_id_with_probe.pk", 'wb') as file:
    #     pickle.dump(cat.algo.ID, file)
    
    # cat.create_cmc_graph()
    # with open("dbfeature_withprobe.pk", 'wb') as file:
    #     pickle.dump(cat.algo.database, file)
    # with open("db_probe_feature.pk", 'wb') as file:
    #     pickle.dump(cat.probe_feature, file)

