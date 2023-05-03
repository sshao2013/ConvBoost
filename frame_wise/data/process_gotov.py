import os

import numpy as np
import pandas as pd
from scipy import stats

NUM_FEATURES = 9


def derive_y(y):
    vals, counts = np.unique(y, return_counts=True)

    vals = sorted(vals)
    print('*' * 30)
    print(vals)
    print(counts)
    print('*' * 30)

    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    inedx_dict = {}
    for i in range(len(vals)):
        inedx_dict[vals[i]] = cls[i]

    for i in range(len(y)):
        y[i] = inedx_dict[y[i]]

    y = y.astype(int)

    return y


def downsampling(data_x, data_y):
    """Recordings are downsamplied to 30Hz, as in the Opportunity dataset

    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """

    idx = np.arange(0, data_x.shape[0], 2)

    return data_x[idx], data_y[idx]


def read_data_gotov(data_dir):
    saved_filename = 'gotov.npy'
    if os.path.isfile(data_dir + saved_filename) == True:
        print(f'[-] Processed File Already Exists, Loading the File')
        data = np.load(data_dir + saved_filename, allow_pickle=True)
    else:
        del_sbj_list = ['GOTOV02', 'GOTOV03', 'GOTOV04', 'GOTOV12', 'GOTOV19', 'GOTOV23']
        validation_sbj_list = ['GOTOV09', 'GOTOV13', 'GOTOV20', 'GOTOV30', 'GOTOV31']
        test_sbj_list = ['GOTOV06', 'GOTOV17', 'GOTOV28', 'GOTOV33', 'GOTOV35']
        del_data = 'equivital'
        traindata_path_list = []
        validdata_path_list = []
        testdata_path_list = []
        # getting the subjects
        for subjects in sorted(os.listdir(data_dir)):

            if subjects in del_sbj_list:
                pass

            else:

                if subjects in validation_sbj_list:
                    sbj = []
                    valid_each_subject_root = data_dir + '/' + subjects
                    for valid_accdata in sorted(os.listdir(valid_each_subject_root)):
                        if del_data in valid_accdata:
                            pass
                        else:
                            valid_each_subject_accdata = valid_each_subject_root + '/' + valid_accdata
                            sbj.append(valid_each_subject_accdata)
                    validdata_path_list.append(sbj)


                elif subjects in test_sbj_list:
                    sbj = []
                    test_each_subject_root = data_dir + '/' + subjects
                    for test_accdata in sorted(os.listdir(test_each_subject_root)):
                        if del_data in test_accdata:
                            pass
                        else:
                            test_each_subject_accdata = test_each_subject_root + '/' + test_accdata
                            sbj.append(test_each_subject_accdata)
                    testdata_path_list.append(sbj)


                else:
                    sbj = []
                    train_each_subject_root = data_dir + subjects
                    for train_accdata in sorted(os.listdir(train_each_subject_root)):
                        if del_data in train_accdata:
                            pass
                        else:
                            train_each_subject_accdata = train_each_subject_root + '/' + train_accdata
                            sbj.append(train_each_subject_accdata)
                    traindata_path_list.append(sbj)

        train_x_data = np.empty((0, 9))
        train_y_data = np.empty((0))

        test_x_data = np.empty((0, 9))
        test_y_data = np.empty((0))

        valid_x_data = np.empty((0, 9))
        valid_y_data = np.empty((0))

        for sbj in traindata_path_list:
            print(sbj)
            path_ankle = sbj[0]
            path_chest = sbj[1]
            path_wrist = sbj[2]
            df_ankle = pd.read_csv(path_ankle)
            df_chest = pd.read_csv(path_chest)
            df_wrist = pd.read_csv(path_wrist)

            data_writst = df_wrist[['x', 'y', 'z']].to_numpy()
            data_ankle = df_ankle[['x', 'y', 'z']].to_numpy()
            data_chest = df_chest[['x', 'y', 'z']].to_numpy()

            label_wrist = df_wrist[['labels']].to_numpy().squeeze()
            label_ankle = df_ankle[['labels']].to_numpy().squeeze()
            label_chest = df_chest[['labels']].to_numpy().squeeze()

            min_len = min([int(label_ankle.shape[0]), int(label_chest.shape[0]), int(label_wrist.shape[0])])
            data_ankle = data_ankle[:min_len]
            data_chest = data_chest[:min_len]
            data_writst = data_writst[:min_len]
            label_wrist = label_wrist[:min_len]
            label_ankle = label_ankle[:min_len]
            label_chest = label_chest[:min_len]

            x = np.concatenate([data_ankle, data_chest, data_writst], axis=1)
            label = np.stack([label_ankle, label_chest, label_wrist]).T
            pp_label = []
            for i in label:
                mode = stats.mode(i)
                pp_label.append(mode[0])
            y = np.array(pp_label).squeeze()

            train_x_data = np.vstack((train_x_data, x))
            train_y_data = np.concatenate([train_y_data, y])

        for sbj in testdata_path_list:
            print(sbj)
            path_ankle = sbj[0]
            path_chest = sbj[1]
            path_wrist = sbj[2]

            df_ankle = pd.read_csv(path_ankle)
            df_chest = pd.read_csv(path_chest)
            df_wrist = pd.read_csv(path_wrist)

            data_writst = df_wrist[['x', 'y', 'z']].to_numpy()
            data_ankle = df_ankle[['x', 'y', 'z']].to_numpy()
            data_chest = df_chest[['x', 'y', 'z']].to_numpy()

            label_wrist = df_wrist[['labels']].to_numpy().squeeze()
            label_ankle = df_ankle[['labels']].to_numpy().squeeze()
            label_chest = df_chest[['labels']].to_numpy().squeeze()

            min_len = min([int(label_ankle.shape[0]), int(label_chest.shape[0]), int(label_wrist.shape[0])])
            data_ankle = data_ankle[:min_len]
            data_chest = data_chest[:min_len]
            data_writst = data_writst[:min_len]
            label_wrist = label_wrist[:min_len]
            label_ankle = label_ankle[:min_len]
            label_chest = label_chest[:min_len]

            x = np.concatenate([data_ankle, data_chest, data_writst], axis=1)
            label = np.stack([label_ankle, label_chest, label_wrist]).T
            pp_label = []
            for i in label:
                mode = stats.mode(i)
                pp_label.append(mode[0])
            y = np.array(pp_label).squeeze()

            test_x_data = np.vstack((test_x_data, x))
            test_y_data = np.concatenate([test_y_data, y])

        for sbj in validdata_path_list:
            print(sbj)
            path_ankle = sbj[0]
            path_chest = sbj[1]
            path_wrist = sbj[2]

            df_ankle = pd.read_csv(path_ankle)
            df_chest = pd.read_csv(path_chest)
            df_wrist = pd.read_csv(path_wrist)

            data_writst = df_wrist[['x', 'y', 'z']].to_numpy()
            data_ankle = df_ankle[['x', 'y', 'z']].to_numpy()
            data_chest = df_chest[['x', 'y', 'z']].to_numpy()

            label_wrist = df_wrist[['labels']].to_numpy().squeeze()
            label_ankle = df_ankle[['labels']].to_numpy().squeeze()
            label_chest = df_chest[['labels']].to_numpy().squeeze()

            min_len = min([int(label_ankle.shape[0]), int(label_chest.shape[0]), int(label_wrist.shape[0])])
            data_ankle = data_ankle[:min_len]
            data_chest = data_chest[:min_len]
            data_writst = data_writst[:min_len]
            label_wrist = label_wrist[:min_len]
            label_ankle = label_ankle[:min_len]
            label_chest = label_chest[:min_len]

            x = np.concatenate([data_ankle, data_chest, data_writst], axis=1)
            label = np.stack([label_ankle, label_chest, label_wrist]).T
            pp_label = []
            for i in label:
                mode = stats.mode(i)
                pp_label.append(mode[0])
            y = np.array(pp_label).squeeze()

            valid_x_data = np.vstack((valid_x_data, x))
            valid_y_data = np.concatenate([valid_y_data, y])

        valid_y_data = valid_y_data.astype(str)
        test_y_data = test_y_data.astype(str)
        train_y_data = train_y_data.astype(str)

        # valid_y_data = derive_y(valid_y_data)
        # test_y_data = derive_y(test_y_data)

        index_nan = np.squeeze(np.array(np.argwhere(train_y_data == 'nan')))
        print(index_nan)
        train_x_data = np.delete(train_x_data, index_nan, axis=0)
        train_y_data = np.delete(train_y_data, index_nan, axis=0)
        train_y_data = derive_y(train_y_data)

        data = {'X_train': train_x_data, 'X_test': test_x_data, 'X_val': valid_x_data, 'y_train': train_y_data,
                'y_test': test_y_data, 'y_val': valid_y_data}
        #
        np.save(os.path.join(data_dir + saved_filename), data)
        data = np.load(data_dir + saved_filename, allow_pickle=True)

    return data
