# -*- coding: utf-8 -*- 
# @Time : 2023/11/23 12:53 
# @Author : DirtyBoy 
# @File : feature_extractor.py
import os
from core.feature.feature_extraction import DrebinFeature, OpcodeSeq, APISequence

if __name__ == '__main__':
    android_features_saving_dir = '../dataset/naive_pool'
    intermediate_data_saving_dir = 'config'

    feature_extractor = OpcodeSeq(android_features_saving_dir, intermediate_data_saving_dir, update=False,
                                    proc_number=8)

    apk_dir = '../dataset/apk'

    mal_feature_list1 = feature_extractor.feature_extraction(apk_dir)
