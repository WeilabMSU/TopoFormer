from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from scipy.stats import pearsonr
from transformers import BatchFeature
import torch
from datasets import Dataset
import numpy as np
import pandas as pd
import os, pickle, glob
import argparse
import sys

from top_transformer import TopTForImageClassification
from top_transformer import TopTForPreTraining


def metrics_func(true_value, predict_value):
    # metrics
    r2 = metrics.r2_score(true_value, predict_value)
    mae = metrics.mean_absolute_error(true_value, predict_value)
    mse = metrics.mean_squared_error(true_value, predict_value)
    rmse = mse ** 0.5
    pearson_r = pearsonr(true_value, predict_value)[0]
    pearson_r2 = pearson_r ** 2

    # print
    print(f"Metric - r2: {r2:.3f} mae: {mae:.3f} mse: {mse:.3f} "
          f"rmse: {rmse:.3f} pearsonr: {pearson_r:.3f} pearsonr2: {pearson_r2:.3f}")
    return r2, mae, mse, rmse, pearson_r, pearson_r2


def get_predictions(
    top_feature_array = None,
    test_label_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/downstream_task_labels/CASF2007_core_test_label.csv',
    scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_large.sav',
    model_path = r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_CASF2007/selected_global_para_32_0.0001_from_3w/model_cls_10000_0',
    save_path = r'/home/chendo11/workfolder/TopTransformer/Output_dir/for_consensus_predict_deep_result_20times/scoring_casf2007/para_32_0.0001.npy'
):

    # data prepare and preprocess
    label_df = pd.read_csv(test_label_file, header=0, index_col=0)
    scaler = pickle.load(open(scaler_path, 'rb'))
    num_sample, num_channel, height, width = np.shape(top_feature_array)
    data_0 = np.reshape(top_feature_array, [num_sample, num_channel*height*width])
    scaled_data = scaler.transform(data_0).reshape([num_sample, num_channel, height, width])
    model_inputs = BatchFeature({"topological_features": scaled_data}, tensor_type='pt')

    # load model
    model = TopTForImageClassification.from_pretrained(model_path)

    # prediction
    with torch.no_grad():
        outputs = model(**model_inputs)
    predicted_value = outputs.logits.squeeze().numpy()
    result_dict = {"predict": predicted_value, "true": label_df.values.squeeze(1)}

    # save result
    np.save(save_path, result_dict, allow_pickle=True)

    # optional, metric
    # metrics_func(result_dict['predict'], result_dict['true'])
    return None


def main_get_predictions_for_scoring():
    def get_top_feature_from_dict(top_feature_file, test_label_file):
        top_feature_dict = np.load(top_feature_file, allow_pickle=True).item()
        label_df = pd.read_csv(test_label_file, header=0, index_col=0)
        top_feature_array = [np.float32(top_feature_dict[key]) for key in label_df.index.tolist()]
        return top_feature_array

    top_feature_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/all_feature_ele_scheme_1-lap0_rips_12-6channel-212-filtration50.npy'
    test_label_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/downstream_task_labels/CASF2016_core_test_label.csv'
    scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav'
    model_pathes = glob.glob(r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_CASF2016_filtration50_212_12/selected_global_para_64_0.00008_from_3w/model_cls_64_0.00008_*')
    for i, model_p in enumerate(model_pathes):
        print(i, model_p)
        if not os.path.exists(os.path.join(model_p, 'all_results.json')):
            continue

        save_path = rf'/home/chendo11/workfolder/TopTransformer/Output_dir/for_consensus_predict_deep_result_20times/scoring_casf2016_212_50/model_para_64_0.00008_{i}.npy'

        get_predictions(
            top_feature_array=get_top_feature_from_dict(top_feature_file, test_label_file),
            test_label_file=test_label_file,
            scaler_path=scaler_path,
            model_path=model_p,
            save_path=save_path,
        )
    

    # # get training predictions
    # top_feature_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/all_feature_ele_scheme_1-lap0_rips_12-6channel-212-filtration50.npy'
    # test_label_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/downstream_task_labels/CASF2007_refine_train_label.csv'
    # scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav'
    # model_pathes = glob.glob(r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_CASF2007_filtration50_212_12/selected_global_para_32_0.00008_from_3w/model_cls_32_0.00008_*')
    # for i, model_p in enumerate(model_pathes):
    #     print(i, model_p)
    #     if not os.path.exists(os.path.join(model_p, 'all_results.json')):
    #         continue

    #     save_path = rf'/home/chendo11/workfolder/TopTransformer/Data_for_different_tasks/Scoring_power/train_set_predictions/casf_2007_trainset_212_50/model_para_32_0.00008_{i}.npy'

    #     get_predictions(
    #         top_feature_array=get_top_feature_from_dict(top_feature_file, test_label_file),
    #         test_label_file=test_label_file,
    #         scaler_path=scaler_path,
    #         model_path=model_p,
    #         save_path=save_path,
    #     )


    return None


def main_get_predictions_for_scoring_2020():
    def get_top_feature_from_dict(top_feature_file, test_label_file):
        top_feature_dict = np.load(top_feature_file, allow_pickle=True).item()
        label_df = pd.read_csv(test_label_file, header=0, index_col=0)
        top_feature_array = [np.float32(top_feature_dict[key]) for key in label_df.index.tolist()]
        return top_feature_array

    top_feature_file = r'TopTransformer/TopFeatures_Data/all_feature_ele_scheme_1-lap0_rips_20-6channel-10.npy'
    test_label_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/downstream_task_labels/CASF2016_core_test_label.csv'
    scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_large.sav'
    model_pathes = glob.glob(r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_v2020_012_20/selected_global_para_64_0.00008_from_3w/model_cls_*')
    for i, model_p in enumerate(model_pathes):
        print(i, model_p)
        if not os.path.exists(os.path.join(model_p, 'all_results.json')):
            continue

        save_path = rf'/home/chendo11/workfolder/TopTransformer/Output_dir/for_consensus_predict_deep_result_20times/scoring_casf2016_from_v2020trained_model/model_para_32_0.00008_{i}.npy'

        get_predictions(
            top_feature_array=get_top_feature_from_dict(top_feature_file, test_label_file),
            test_label_file=test_label_file,
            scaler_path=scaler_path,
            model_path=model_p,
            save_path=save_path,
        )
    return None


def main_get_predictions_for_docking():
    
    feature_folder = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/for-CASF-200713-dock-features-212-12'
    scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav'
    label_folder = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/for-CASF-200713-dock-labels'
    all_model_folder = r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_docking_all_train'

    # # label 2007
    # target_label_files = [ll for ll in glob.glob(os.path.join(label_folder, '*.csv')) if '.rms_' in ll]
    # save_folder = r'/home/chendo11/workfolder/TopTransformer/Data_for_different_tasks/Docking_power/final_casf2007_results'
    
    # label 2013
    target_label_files = [ll for ll in glob.glob(os.path.join(label_folder, '*.csv')) if '_rmsd.dat' in ll]
    save_folder = r'/home/chendo11/workfolder/TopTransformer/Data_for_different_tasks/Docking_power/final_casf2013_results'


    # get docking predicted result for all
    for i, label_f in enumerate(target_label_files):
        print(i, label_f)

        # prepare the input data
        data_name = os.path.split(label_f)[-1].split('.csv')[0]
        top_feature_file = os.path.join(feature_folder, f'{data_name}.npy')
        top_feature_array = np.load(top_feature_file, allow_pickle=True)[:, :, 0::2, :].astype('float32')

        model_path_temp = os.path.join(all_model_folder, f'{data_name[0:4]}.data_train*')
        model_p = glob.glob(model_path_temp)[0]

        save_path = os.path.join(save_folder, f'{data_name[0:4]}.predictions.npy')

        get_predictions(
            top_feature_array=top_feature_array,
            test_label_file=label_f,
            scaler_path=scaler_path,
            model_path=model_p,
            save_path=save_path,
        )
    
    return None


def main():
    # get_predictions()
    # main_get_predictions_for_scoring()
    main_get_predictions_for_scoring_2020()
    # main_get_predictions_for_docking()
    return None


if __name__ == "__main__":
    main()
    print('End!')