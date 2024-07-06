"""Summary:
        Generate the latent features from pretrained model.
    
    Author:
        Dong Chen
    Creat:
        05-24-2024
    Last modify:
        05-24-2024
    Dependencies:
        python                    3.9.12
        torch                     1.13.0.dev20221006+cu117
        transformers              4.23.1
    Note:
        - Helpful document https://huggingface.co/docs/transformers/v4.26.1/en/model_doc/vit#transformers.ViTModel
        - 
"""


from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from transformers import BatchFeature
import torch
from datasets import Dataset
import numpy as np
import pandas as pd
import os, pickle
import argparse
import sys

from top_transformer import TopTModel
from top_transformer import TopTForPreTraining
from top_transformer import TopTForImageClassification


def pooler_func(pooler_type):
    if pooler_type == 'avg':
        feature_pooler = lambda x: torch.mean(x, dim=1)
    elif pooler_type == 'bos':
        feature_pooler = lambda x: x[:, 0, :]
    else:  # default is avg
        feature_pooler = lambda x: torch.mean(x, dim=1)
    return feature_pooler


def main_latent_from_pretrained_model(args):
    # prepare input
    scaler_path = args.scaler_path
    model_path = args.model_path
    feature_path = args.feature_path
    top_feature_array = np.load(feature_path, allow_pickle=True)
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = TopTModel.from_pretrained(model_path)
    pooler = pooler_func(pooler_type=args.pooler_type)

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Extract feature on {device} device.')
    
    # scale
    num_sample, num_channel, height, width = np.shape(top_feature_array)
    print(f'Input shape: ({num_sample}, {num_channel}, {height}, {width})')
    data_0 = np.reshape(top_feature_array, [num_sample, num_channel*height*width])
    scaled_data = scaler.transform(data_0).reshape([num_sample, num_channel, height, width])
    model_inputs = BatchFeature({"topological_features": scaled_data}, tensor_type='pt').to(device)
    
    # model
    model.to(device)
    model.config.update(dict(mask_ratio=0))

    # get latent
    model.eval()
    with torch.no_grad():
        outputs = model(**model_inputs)
    latent_features = pooler(outputs.last_hidden_state).cpu().numpy()
    print('Output shape:', np.shape(latent_features))

    # save feature
    save_feature_path = args.save_feature_path
    np.save(save_feature_path, latent_features.astype(np.float32), allow_pickle=True)

    return None


def main_latent_from_finetuned_model(args):
    # prepare input
    scaler_path = args.scaler_path
    model_path = args.model_path
    feature_path = args.feature_path
    top_feature_array = np.load(feature_path, allow_pickle=True)
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = TopTForImageClassification.from_pretrained(model_path)
    pooler = pooler_func(pooler_type=args.pooler_type)


    # device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Extract feature on {device} device.')
    
    # scale
    num_sample, num_channel, height, width = np.shape(top_feature_array)
    print(f'Input shape: ({num_sample}, {num_channel}, {height}, {width})')
    data_0 = np.reshape(top_feature_array, [num_sample, num_channel*height*width])
    scaled_data = scaler.transform(data_0).reshape([num_sample, num_channel, height, width])
    model_inputs = BatchFeature({"topological_features": scaled_data}, tensor_type='pt').to(device)
    
    # model
    model.to(device)
    model.config.update(dict(mask_ratio=0))

    # get latent
    model.eval()
    with torch.no_grad():
        outputs = model(output_hidden_states=True, **model_inputs)
    latent_features = pooler(outputs.hidden_states[-1]).cpu().numpy()
    print('Output shape:', np.shape(latent_features))

    # save feature
    save_feature_path = args.save_feature_path
    np.save(save_feature_path, latent_features.astype(np.float32), allow_pickle=True)

    return None


def main_latent_from_pretrained_decoder(args):
    # prepare input
    scaler_path = args.scaler_path
    model_path = args.model_path
    feature_path = args.feature_path
    top_feature_array = np.load(feature_path, allow_pickle=True)
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = TopTForPreTraining.from_pretrained(model_path)
    pooler = pooler_func(pooler_type=args.pooler_type)

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Extract feature on {device} device.')
    
    # scale
    num_sample, num_channel, height, width = np.shape(top_feature_array)
    print(f'Input shape: ({num_sample}, {num_channel}, {height}, {width})')
    data_0 = np.reshape(top_feature_array, [num_sample, num_channel*height*width])
    scaled_data = scaler.transform(data_0).reshape([num_sample, num_channel, height, width])
    model_inputs = BatchFeature({"topological_features": scaled_data}, tensor_type='pt').to(device)
    
    # model
    model.to(device)
    model.config.update(dict(mask_ratio=0))

    # get latent
    model.eval()
    with torch.no_grad():
        eocoder_outputs = model.topt(**model_inputs)
        latent = eocoder_outputs.last_hidden_state
        ids_restore = eocoder_outputs.ids_restore
        mask = eocoder_outputs.mask
        outputs = model.decoder(latent, ids_restore, output_hidden_states=True)
    latent_features = pooler(outputs.hidden_states[-1]).cpu().numpy()
    print('Output shape:', np.shape(latent_features))

    # save feature
    save_feature_path = args.save_feature_path
    np.save(save_feature_path, latent_features.astype(np.float32), allow_pickle=True)

    return None


def parse_args(args):
    parser = argparse.ArgumentParser(description="extract latent features")

    parser.add_argument('--model_path', default="pretrain_alldata_mask20_6channels_large", type=str, help='The folder contain the checkpoints')
    parser.add_argument('--scaler_path', default="pretrain_data_standard_minmax_6channel.sav", type=str)
    parser.add_argument('--feature_path', default="rips_20-6channel-10.npy", type=str,
                        help='The feature file path (.npy format), containing the numpy array (not in dict)')
    parser.add_argument('--save_feature_path', default="latent_feature.sav", type=str)
    parser.add_argument('--pooler_type', default="avg", type=str, help='select from {avg: average, bos: embeding of first filtration parameter}')
    parser.add_argument('--latent_type', default="encoder_pretrain", type=str, help='select from {encoder_pretrain: encoder of pretrained model, encoder_finetuned: encoder of finetuned model, decoder_pretrain: decoder of the pretrained model}')
    
    args = parser.parse_args()
    return args


def main_cli():
    args = parse_args(sys.argv[1:])
    print(args)
    if args.latent_type == 'encoder_pretrain':
        main_latent_from_pretrained_model(args)
    elif args.latent_type == 'encoder_finetuned':
        main_latent_from_finetuned_model(args)
    elif args.latent_type == 'decoder_pretrain':
        main_latent_from_pretrained_decoder(args)


if __name__ == "__main__":
    main_cli()
