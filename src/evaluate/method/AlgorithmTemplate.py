# -*- coding: utf-8 -*-

class AlgorithmTemplate:
    def __init__(self):
        print("init")
    def feature_extract(self,image_path):
        print("feature_extract")
        template = "feature"
        return template
    
    def match_1to1(self,template1,template2):
        score = 1
        return score
    
    def enroll_to_DB(self,template,_id):
        print("enroll")
    
    def retriev(self,num_result):
        listt = ["topmatch1_id","topmatch2_id","topmatch3_id"]
        score = [0.8,0.7,0.6]
        return (listt,score)
   