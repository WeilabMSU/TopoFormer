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


def get_attention_weights_for_docking():

    # prepare_data
    # top_feature_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/for-CASF-200713-dock-features-212-12/1a30_rmsd.dat_test.npy'
    top_feature_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/for-CASF-200713-dock-features-212-12/1ajq.rms_test.npy'
    top_feature_array = np.load(top_feature_file, allow_pickle=True)[:, :, 0::2, :].astype('float32')
    print(np.shape(top_feature_array))
    num_sample, num_channel, height, width = np.shape(top_feature_array)
    data_0 = np.reshape(top_feature_array, [num_sample, num_channel*height*width])

    # date preprocess
    scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav'
    scaler = pickle.load(open(scaler_path, 'rb'))
    scaled_data = scaler.transform(data_0).reshape([num_sample, num_channel, height, width])

    # load model
    # model_path = r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_docking_all_train/1a30.data_train'
    model_path = r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_docking_all_train/1ajq.data_train'
    model = TopTForImageClassification.from_pretrained(model_path)
    model.eval()
    SEED = 1242
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # model input
    # test_label_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/for-CASF-200713-dock-labels/1ajq_rmsd.dat_test.csv'
    # rmsd_0 = scaled_data[-1::, :, :, :]
    # rmsd_1_33 = scaled_data[-8:-7, :, :, :]
    # rmsd_3_96 = scaled_data[-4:-3, :, :, :]
    # rmsd_7_16 = scaled_data[-2:-1, :, :, :]
    # rmsd_9_06 = scaled_data[74:75, :, :, :]

    rmsd_0 = scaled_data[-1::, :, :, :]
    rmsd_1_33 = scaled_data[10:11, :, :, :]
    rmsd_3_96 = scaled_data[50:51, :, :, :]
    rmsd_7_16 = scaled_data[60:61, :, :, :]
    rmsd_9_06 = scaled_data[70:71, :, :, :]

    save_score = []
    import matplotlib.pyplot as plt
    for i, rmsd_d in enumerate([rmsd_0, rmsd_1_33, rmsd_3_96, rmsd_7_16, rmsd_9_06]):

        model_inputs = BatchFeature({"topological_features": rmsd_d}, tensor_type='pt')
        with torch.no_grad():
            outputs = model(**model_inputs, output_attentions=True)
        attentions = outputs.attentions
        print(len(attentions), np.shape(attentions[0]))

        # Average attention weights across all heads and layers
        avg_attention = sum(att.mean(dim=1) for att in attentions) / len(attentions)
        
        # only the first patch is used to predict score, so use the 1st column
        imdata = avg_attention[0, :, :].detach().numpy()[1::, 0]

        save_score.append(imdata)

        plt.plot(imdata, label=i, color=f'C{i}', alpha=0.4)
        plt.plot(np.argsort(imdata)[-1], np.sort(imdata)[-1], marker='o', color=f'C{i}')
        # plt.imshow(imdata)
    plt.legend()
    plt.savefig(f'attentio-1.png')

    np.save('/home/chendo11/workfolder/TopTransformer/plot_pictures/fig_filtration_importance/1ajq_filtration_score.npy', save_score, allow_pickle=True)
    return None


def get_saliency_maps_for_screening():

    # prepare_data for true ligand
    top_feature_file = r'/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/all_feature_ele_scheme_1-lap0_rips_12-6channel-212-filtration50.npy'
    top_feature_array = np.array([np.load(top_feature_file, allow_pickle=True).item()['1e66']]).astype('float32')
    print(np.shape(top_feature_array))
    num_sample, num_channel, height, width = np.shape(top_feature_array)
    data_0 = np.reshape(top_feature_array, [num_sample, num_channel*height*width])

    # date preprocess
    scaler_path = r'/home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_6channel_filtration50-12.sav'
    scaler = pickle.load(open(scaler_path, 'rb'))
    scaled_data = scaler.transform(data_0).reshape([num_sample, num_channel, height, width])

    # load model
    model_path = r'/home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_CASF2013_filtration50_212_12/for_screening_finetune_from_3w_addv2015_0.1/1e66_train.data_train'
    model = TopTForImageClassification.from_pretrained(model_path)
    # model.eval()
    SEED = 1242
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # model input
    label = np.array([5.37]).astype('float32')

    # maps
    model_inputs = BatchFeature({"topological_features": scaled_data, 'labels': label}, tensor_type='pt')
    model_inputs["topological_features"] = model_inputs["topological_features"].clone().detach().requires_grad_(True)
    # with torch.no_grad():
    outputs = model(**model_inputs)

    loss = outputs.loss  # Assuming you have a loss if you're using BertForSequenceClassification, otherwise just use your regression output
    loss.backward()

    # Get gradients
    input_id_gradients = model_inputs["topological_features"].grad.detach().numpy()[0, :, :, :]
    print(np.shape(input_id_gradients))

    input_id_gradients = np.abs(np.mean(input_id_gradients, axis=0))

    # def softmax(x):
    #     e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    #     return e_x / e_x.sum(axis=0, keepdims=True)
    # input_id_gradients = softmax(input_id_gradients)

    # save data
    np.save(
        r'/home/chendo11/workfolder/TopTransformer/plot_pictures/fig_saliency_maps/1e66_avg_channel_gradients.npy',
        input_id_gradients, allow_pickle=True
    )

    # save picture
    import matplotlib.pyplot as plt
    plt.imshow(input_id_gradients, cmap='Blues')
    plt.savefig(f"grad_0.png")
    plt.imshow(np.sum(scaled_data[0, :, :, :], axis=0), cmap='Blues')
    plt.savefig(f"grad_22.png")

    return None


def main():
    # get_attention_weights_for_docking()
    get_saliency_maps_for_screening()
    return None


if __name__ == "__main__":
    main()
