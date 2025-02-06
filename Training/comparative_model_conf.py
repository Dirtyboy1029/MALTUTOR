# -*- coding: utf-8 -*- 
# @Time : 2024/7/24 11:15 
# @Author : DirtyBoy 
# @File : comparative_model_conf.py
import os, random, time
import numpy as np
import pandas as pd

random.seed(int(time.time()))


def balance_dataset(malware_dict):
    counts = [len(images) for images in malware_dict.values()]
    target_count = int(np.mean(counts))
    balanced_dict = {}

    for family, malware in malware_dict.items():
        if len(malware) >= target_count:
            balanced_dict[family] = random.sample(malware, target_count)
        elif len(malware) == 1:
            balanced_dict[family] = malware + random.choices(malware, k=target_count - len(malware))
        else:
            copies_needed = target_count - len(malware)
            extended_images = malware * copies_needed
            try:
                balanced_dict[family] = random.sample(extended_images, target_count)
            except ValueError:
                balanced_dict[family] = random.choices(extended_images, k=target_count)

    return balanced_dict


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_data_type', type=str, default="malradar")
    parser.add_argument('-comparative_type', '-ct', type=str, default="augment", choices=['sampling', 'augment'])
    args = parser.parse_args()
    train_data_type = args.train_data_type
    comparative_type = args.comparative_type

    if comparative_type == 'sampling':
        data_filenames, labels = read_joblib('config/' + train_data_type + '_database.drebin')
        data_filenames = [item.replace('.drebin', '') for item in data_filenames]

        data_malware = data_filenames[:np.sum(labels)]
        data_benign = data_filenames[np.sum(labels):]

        family_path = '../dataset/family_source_file/' + train_data_type + '_family.csv'
        drebin_family = pd.read_csv(family_path)
        hash_set = set(data_filenames)

        drebin_family = drebin_family[drebin_family['sha256'].isin(hash_set)]
        print(drebin_family.shape)

        malware_family = [drebin_family[drebin_family['sha256'] == h]['family'].values[0] if not drebin_family[
            drebin_family['sha256'] == h].empty else None for h in data_malware]

        malware_family_df = pd.DataFrame({'sha256': data_malware,
                                          'family': malware_family})

        family_hash_dict = malware_family_df.groupby('family')['sha256'].apply(list).to_dict()
        balanced_dict = balance_dataset(family_hash_dict)
        malware_hash = []
        for _, malware in balanced_dict.items():
            malware_hash = malware_hash + malware
        malware_hash = [item + '.drebin' for item in malware_hash]
        benign_hash = [item + '.drebin' for item in data_benign]
        for i in range(3):
            benign_hash = random.sample(benign_hash, k=len(malware_hash))
            save_path = 'config/' + train_data_type + '_database_sampling'+str(i)+'.drebin'
            data_filenames = malware_hash + benign_hash
            labels = np.array([1] * len(malware_hash) + [0] * len(benign_hash))

            dump_joblib((data_filenames, labels), save_path)

