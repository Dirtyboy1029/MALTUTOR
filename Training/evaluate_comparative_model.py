# -*- coding: utf-8 -*- 
# @Time : 2024/6/1 21:32 
# @Author : DirtyBoy 
# @File : evaluate_comparative_model.py

from core.data_preprocessing import data_preprocessing_for_test, data_preprocessing_for_ood
from core.ensemble.vanilla import Vanilla
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-evaluate_type', type=str, default="ood", choices=['ood', 'dataset'])
    parser.add_argument('-test_data_type', type=str, default="drebin")
    parser.add_argument('-train_data_type', type=str, default="malradar")
    parser.add_argument('-comparative_type', '-ct', type=str, default="sampling")
    args = parser.parse_args()
    comparative_type = args.comparative_type
    evaluate_type = args.evaluate_type
    test_data_type = args.test_data_type
    data_type = args.train_data_type
    if comparative_type == 'sampling' or comparative_type == 'smote':
        for random_index in range(3):
            if comparative_type == 'sampling':
                Data_type = data_type + '_' + comparative_type + str(random_index)
            else:
                Data_type = data_type
            print('********************************************************************************')
            print(
                '********evaluate ' + test_data_type + ' set on model which trainset is ' + data_type + '_' + comparative_type + str(
                    random_index) + ' set ********')
            print('********************************************************************************')
            if evaluate_type == 'dataset':
                dataset, gt_labels, _, _, _ = data_preprocessing_for_test(
                    feature_type='drebin', train_data_type=Data_type,
                    data_type=test_data_type)
            elif evaluate_type == 'ood':
                dataset, gt_labels, input_dim, _, data_filenames = data_preprocessing_for_ood(
                    feature_type='drebin', train_data_type=Data_type,
                    data_type=test_data_type)
            else:
                dataset = None
                gt_labels = None
            print('--------------------------------' + str(random_index) + '--------------------------------')
            source_path = '../Model/' + comparative_type + '_model/' + data_type + '_' + comparative_type + str(
                random_index) + '/drebin'
            source_model = Vanilla(architecture_type='dnn',
                                   model_directory=source_path)
            source_model.evaluate(dataset, gt_labels)
    else:
        print('********************************************************************************')
        print(
            '********evaluate ' + test_data_type + ' set on model which trainset is ' + data_type + '_' + comparative_type + ' set ********')
        print('********************************************************************************')
        if evaluate_type == 'dataset':
            dataset, gt_labels, _, _, _ = data_preprocessing_for_test(
                feature_type='drebin', train_data_type=data_type,
                data_type=test_data_type)
        elif evaluate_type == 'ood':
            dataset, gt_labels, input_dim, _, data_filenames = data_preprocessing_for_ood(
                feature_type='drebin', train_data_type=data_type,
                data_type=test_data_type)
        else:
            dataset = None
            gt_labels = None
        source_path = '../Model/' + comparative_type + '_model/' + data_type + '_' + comparative_type + '/drebin'
        source_model = Vanilla(architecture_type='dnn',
                               model_directory=source_path)
        source_model.evaluate(dataset, gt_labels)
