# -*- coding: utf-8 -*- 
# @Time : 2024/7/24 15:52 
# @Author : DirtyBoy 
# @File : training_comparative_model.py
from core.data_preprocessing import data_preprocessing
from core.ensemble.vanilla import Vanilla
from core.dataset_lib import build_dataset_from_numerical_data
import argparse, random
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_type', type=str, default="malradar")
    parser.add_argument('-comparative_type', '-ct', type=str, default="smote",
                        choices=['sampling', 'smote', 'weight', 'csl'])
    args = parser.parse_args()
    train_data_type = args.train_data_type
    comparative_type = args.comparative_type
    feature_type = 'drebin'
    if comparative_type == 'sampling':
        for i in range(3):
            data_type = train_data_type + '_' + comparative_type + str(i)
            dataset, gt_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
                feature_type=feature_type, data_type=data_type)
            vanilla = Vanilla(architecture_type='dnn',
                              model_directory="../Model/" + comparative_type + "_model/" + data_type + '/' + feature_type)
            vanilla_prob, vanilla_training_log = vanilla.fit(train_set=dataset, validation_set=dataset,
                                                             input_dim=input_dim,
                                                             EPOCH=30,
                                                             test_data=dataX_np,
                                                             training_predict=False)
            vanilla.save_ensemble_weights()
    elif comparative_type == 'smote':
        data_type = train_data_type
        dataset, gt_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
            feature_type=feature_type, data_type=data_type)
        malware_index = np.where(gt_labels == 1)[0]
        benware_index = np.where(gt_labels == 0)[0]
        malware_data_filenames = [data_filenames[index].split('.')[0] for index in malware_index]
        malware_dataX_np = dataX_np[malware_index]
        benware_dataX_np = dataX_np[benware_index]

        family_label = pd.read_csv('../dataset/family_source_file/' + data_type + '_family.csv')
        y_family_label = []
        for i in range(len(malware_data_filenames)):
            try:
                y_family_label.append(
                    family_label.loc[family_label['sha256'] == malware_data_filenames[i]]['family'].tolist()[0])
            except Exception:
                malware_dataX_np = np.delete(malware_dataX_np, i, axis=0)
        y_family_label = np.array(y_family_label)
        from collections import defaultdict

        threshold = 5
        my_dict = defaultdict(list)
        for index, value in enumerate(y_family_label):
            my_dict[value].append(index)
        tmp_remove_indexs = []
        tmp_save_indexs = []
        for key, value in my_dict.items():
            if len(value) <= threshold:
                tmp_remove_indexs = tmp_remove_indexs + value
            else:
                tmp_save_indexs = tmp_save_indexs + value

        malware_dataX_np_save = malware_dataX_np[tmp_save_indexs]
        malware_dataX_np_move = malware_dataX_np[tmp_remove_indexs]
        y = []
        for index in tmp_save_indexs:
            y.append(y_family_label[index])

        import time

        start_time = time.time()

        smote = SMOTE(random_state=42, k_neighbors=threshold, sampling_strategy='auto')

        X_resampled, y_resampled = smote.fit_resample(malware_dataX_np_save, y)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"SMOTE Over-Sampling : {execution_time:.2f} s")
        for i in range(3):
            random.seed(i)
            random_indices = np.random.choice(X_resampled.shape[0], size=len(tmp_save_indexs), replace=False)
            random_sample = X_resampled[random_indices]
            combined_array = np.concatenate((random_sample, malware_dataX_np_move), axis=0)
            dataX_np = np.concatenate((combined_array, benware_dataX_np), axis=0)
            gt_labels = np.array([1] * len(combined_array) + [0] * len(benware_dataX_np))
            tmp_dataset = build_dataset_from_numerical_data((dataX_np, gt_labels))
            vanilla = Vanilla(architecture_type='dnn',
                              model_directory="../Model/" + comparative_type + "_model/" + data_type + '_smote' + str(
                                  i) + '/' + feature_type)
            vanilla_prob, vanilla_training_log = vanilla.fit(train_set=tmp_dataset, validation_set=tmp_dataset,
                                                             input_dim=input_dim,
                                                             EPOCH=30,
                                                             test_data=dataX_np,
                                                             training_predict=False)
            vanilla.save_ensemble_weights()
    elif comparative_type == 'weight' or comparative_type == 'csl':

        def generate_family_weights(train_data_type):
            malware_index = np.where(gt_labels == 1)[0]
            malware_data_filenames = [data_filenames[index].split('.')[0] for index in malware_index]
            family_label = pd.read_csv('../dataset/family_source_file/' + train_data_type + '_family.csv')
            y_family_label = []
            for i in range(len(malware_data_filenames)):
                try:
                    y_family_label.append(
                        family_label.loc[family_label['sha256'] == malware_data_filenames[i]]['family'].tolist()[0])
                except Exception:
                    y_family_label.append('unknown')
            classes = np.unique(y_family_label)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_family_label)
            class_weight_dict = dict(zip(classes, class_weights))

            malware_samples_weight = []
            for item in y_family_label:
                malware_samples_weight.append(class_weight_dict[item])
            benign_samples_weight = [1.] * len(np.where(gt_labels == 0)[0])
            samples_weight = np.array(malware_samples_weight + benign_samples_weight)
            return samples_weight


        def generate_uncertainty_weights(train_data_type):

            def sort_indices_descending(lst):
                sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=False)
                return sorted_indices

            def load_curriculum(data_type, n_clusters=3, val_type='self', feature_type='drebin'):
                classifier_data = pd.read_csv(
                    '../dataset_reconstruction/inter_file/' + data_type + '_' + feature_type + '_' + val_type + '_' + str(
                        n_clusters) + '.csv')
                malware = classifier_data[classifier_data['gt_label'] == 1]
                curriculum = pd.read_csv(
                    '../dataset_reconstruction/inter_file/' + data_type + '_' + feature_type + '_' + val_type + '_' + str(
                        n_clusters) + '.csv')[
                                 'label'].tolist()[0:malware.shape[0]]
                malware_df_set = [malware[malware['label'] == i] for i in range(n_clusters)]
                uc_data = pd.read_csv(
                    '../dataset_reconstruction/uc_metrics_csv/' + data_type + '_' + feature_type + '_' + val_type + '_val.csv')[
                    ['apk_name', 'no_kld_30', 'with_kld_label_30']]
                malware_df_set = [pd.merge(item, uc_data, on='apk_name', how='inner') for item in malware_df_set]

                malware_fitting_degree = sort_indices_descending(
                    [np.sum(np.sqrt(item['no_kld_30'] ** 2 + item['with_kld_label_30'] ** 2)) / item.shape[0] for item
                     in
                     malware_df_set])

                curriculum_ = []
                for i in range(n_clusters):
                    tmp = []
                    for j, index in enumerate(curriculum):
                        if index == malware_fitting_degree[i]:
                            tmp.append(j)
                    curriculum_.append(tmp)
                return curriculum_

            curriculum = load_curriculum(train_data_type)

            malware_samples_weight = np.zeros(np.sum(gt_labels))
            for item in curriculum[0]:
                malware_samples_weight[item] = 0.01
            for item in curriculum[1]:
                malware_samples_weight[item] = 1
            for item in curriculum[2]:
                malware_samples_weight[item] = 100
            benign_samples_weight = [1.] * len(np.where(gt_labels == 0)[0])
            samples_weight = np.array(list(malware_samples_weight) + benign_samples_weight)
            return samples_weight


        dataset, gt_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
            feature_type=feature_type, data_type=train_data_type)
        if comparative_type == 'weight':
            generate_weights = generate_family_weights
        elif comparative_type == 'csl':
            generate_weights = generate_uncertainty_weights
        else:
            generate_weights = None
        samples_weight = generate_weights(train_data_type)

        trainset = build_dataset_from_numerical_data((dataX_np, gt_labels, samples_weight))

        vanilla = Vanilla(architecture_type='dnn',
                          model_directory="../Model/" + comparative_type + "_model/" + train_data_type + '_' + comparative_type + '/' + feature_type)
        vanilla_prob, vanilla_training_log = vanilla.fit(train_set=trainset, validation_set=dataset,
                                                         input_dim=input_dim,
                                                         EPOCH=30,
                                                         training_predict=False)
        vanilla.save_ensemble_weights()
