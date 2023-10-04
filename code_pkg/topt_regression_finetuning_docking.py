"""Summary:
        Finetuning script
    
    Author:
        Dong Chen
    Creat:
        03-09-2023
    Last modify:
        03-09-2022
    Dependencies:
        python                    3.9.12
        torch                     1.13.0.dev20221006+cu117
        transformers              4.23.1
    Note:
        - Helpful document https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining
        - 
"""


import logging, sys
from typing import Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    BatchFeature,
)
from datasets import Dataset
from datasets import load_metric
from dataclasses import dataclass, field
import torch, os, pickle
import numpy as np
import pandas as pd
import glob

from top_transformer import TopTConfig
from top_transformer import TopTForImageClassification


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="pdb2020", metadata={"help": "Name of a dataset"}
    )
    scaler_path: Optional[str] = field(default=None, metadata={"help": "scaler path"})
    feature_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of the images in the files."}
    )
    train_data: Optional[str] = field(
        default='/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/top_v2016_refine_train_ele_scheme_1-norm_ph_vr-10.npy',
        metadata={"help": "A folder containing the training data, .npy file."}
    )
    train_label: Optional[str] = field(
        default='/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/v2016_refine_train_label.csv',
        metadata={"help": "A folder containing the training data, .csv file."}
    )
    validation_data: Optional[str] = field(default='/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/top_v2016_core_test_ele_scheme_1-norm_ph_vr-10.npy', metadata={"help": "A folder containing the validation data."})
    validation_label: Optional[str] = field(default='/home/chendo11/workfolder/TopTransformer/TopFeatures_Data/v2016_core_test_label.csv', metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.0, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    random_seed: Optional[int] = field(default=12345)

    def __post_init__(self):
        data_files = {}
        if self.train_data is not None:
            data_files["train"] = self.train_data
        if self.validation_data is not None:
            data_files["val"] = self.validation_data
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default='/home/chendo11/workfolder/TopTransformer/Output_dir/pretrain_from_vit_gpu',
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    mask_ratio: float = field(
        default=0.0, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )
    num_labels: int = field(
        default=1, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )
    hidden_dropout_prob: float = field(default=0.1)
    attention_probs_dropout_prob: float = field(default=0.1)
    num_channels: int = field(default=3)
    loss_on_patches: str = field(default='on_removed_patches') # on_all_patches
    pooler_type: str = field(default='cls_token')
    specify_loss_fct: str = field(default='mse')


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=8e-5, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )
    output_dir: Optional[str] = field(
        default='/home/chendo11/workfolder/TopTransformer/Output_dir/finetuning_for_v2016',
        metadata={"help": "A folder to save the pretrianed model"}
    )
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Training a new model to overwrite the exist dir"})
    warmup_steps: float = field(default=0.1)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=128)
    num_train_epochs: int = field(default=100)
    save_strategy: str = field(default="epoch")
    evaluation_strategy: str = field(default="epoch")
    # save_steps: int = field(default=1000)
    save_total_limit: int = field(default=2)
    no_cuda: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    load_best_model_at_end: bool = field(default=True)
    max_steps: int = field(default=10000)
    seed: int = field(default=12345)


def collate_fn(examples):
    topological_features = torch.stack([torch.as_tensor(example["topological_features"]) for example in examples])
    return {"topological_features": topological_features, 'labels': torch.tensor([x['labels'] for x in examples])}


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


def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def feature_extractor(input_features):
    return BatchFeature(input_features, tensor_type='pt')


def scaler_for_image_like_data(data, scaler=None):
    """data size = [num_sample, num_channel, height, width]"""
    num_sample, num_channel, height, width = np.shape(data)
    data_0 = np.reshape(data, [num_sample, num_channel*height*width])
    if scaler is None:
        scaler = Pipeline([
            ('scaler1', StandardScaler()),
            ('scaler2', MinMaxScaler())
        ])
        # scaler = RobustScaler()
        scaler.fit(data_0)
        pickle.dump(scaler, open(
            os.path.join(os.path.split(os.path.realpath(__file__))[0],'pretrain_data_standard_minmax_3channel.sav'), 'wb'))
    data_1 = scaler.transform(data_0)
    out_data = np.reshape(data_1, [num_sample, num_channel, height, width])
    return out_data, scaler


def TopT_Finetuning():
    logger = logging.getLogger(__name__)

    # arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # load data
    if data_args.scaler_path is None:
        scaler_path = os.path.join(os.path.split(os.path.realpath(__file__))[0],'pretrain_data_standard_minmax.sav')
    else:
        scaler_path = data_args.scaler_path
    if os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, 'rb'))
    else:
        scaler = None
    
    if data_args.train_data is not None:
        train_label_df = pd.read_csv(data_args.train_label, header=0, index_col=0)
        train_list = list(train_label_df.index)
        train_label = train_label_df.loc[train_list].values

        train_file_to_data = np.load(data_args.train_data, allow_pickle=True)[:, :, 0::2, :]
        train_file_to_data, scaler = scaler_for_image_like_data(train_file_to_data, scaler)

        ds_train = Dataset.from_dict({'topological_features': train_file_to_data, 'labels': train_label})

    if data_args.train_data is None and not os.path.exists(scaler_path):
        raise ValueError('No scaler and train_data. Make sure all data should be standardized')

    if data_args.validation_data == "None":
        data_args.validation_data = None

    if data_args.validation_data is not None:
        valid_label_df = pd.read_csv(data_args.validation_label, header=0, index_col=0)
        valid_list = list(valid_label_df.index)
        valid_label = valid_label_df.loc[valid_list].values

        valid_file_to_data = np.load(data_args.validation_data, allow_pickle=True)
        valid_file_to_data, scaler = scaler_for_image_like_data(valid_file_to_data, scaler)

        ds_valid = Dataset.from_dict({'topological_features': valid_file_to_data, 'labels': valid_label})
    elif isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds_train.train_test_split(data_args.train_val_split)
        ds_train = split['train']
        ds_valid = split['test']

    # do train, prepare the data
    def preprocess_features(example_batch):
        """Preprocess a batch of images by applying transforms."""
        example_batch["topological_features"] = [image for image in example_batch['topological_features']]
        example_batch['labels'] = example_batch['labels']
        return example_batch

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            ds_train = ds_train.shuffle(seed=data_args.random_seed).select(range(data_args.max_train_samples))
        else:
            ds_train = ds_train.shuffle(seed=data_args.random_seed)
        # ds_train.set_transform(preprocess_features)
    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            ds_valid = ds_valid.shuffle(seed=data_args.random_seed).select(range(data_args.max_eval_samples))
        else:
            ds_valid = ds_valid.shuffle(seed=data_args.random_seed)
        # ds_valid.set_transform(preprocess_features)

    # create the model
    if model_args.model_name_or_path:
        config = TopTConfig.from_pretrained(model_args.model_name_or_path)
        config.update(dict(
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
            num_labels=model_args.num_labels,
            mask_ratio=model_args.mask_ratio,
            pooler_type=model_args.pooler_type,
            specify_loss_fct=model_args.specify_loss_fct,
        ))
        model = TopTForImageClassification.from_pretrained(model_args.model_name_or_path, config=config)

    # batch size and absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256
        # training_args.learning_rate = training_args.base_learning_rate

    # Initializer the trainer
    print('start to train')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train if training_args.do_train else None,
        eval_dataset=ds_valid if training_args.do_eval else None,
        # tokenizer=feature_extractor,
        data_collator=collate_fn,
        # compute_metrics=compute_metrics,
    )

    # Training
    print("build trainer with on device:", training_args.device, "with n gpus:", training_args.n_gpu)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return None


def main():
    """
        Normal example:
        CUDA_VISIBLE_DEVICES=2 python /home/chendo11/workfolder/TopTransformer/code_pkg/topt_regression_finetuning.py --hidden_dropout_prob 0.5 --attention_probs_dropout_prob 0.1 --num_train_epochs 50 --per_device_train_batch_size 64 --output_dir /home/chendo11/workfolder/TopTransformer/Output_dir/finetune_for_regression_2 --model_name_or_path /home/chendo11/workfolder/TopTransformer/Output_dir/pretrain_allpdb_mask30_3channels/checkpoint-11590 --scaler_path /home/chendo11/workfolder/TopTransformer/code_pkg/pretrain_data_standard_minmax_3channel.sav --validation_data /home/chendo11/workfolder/TopTransformer/TopFeatures_Data/top_v2016_core_test_ele_scheme_1-combined_3channel-10.npy --train_data /home/chendo11/workfolder/TopTransformer/TopFeatures_Data/top_v2016_refine_train_ele_scheme_1-combined_3channel-10.npy
        
         --max_train_samples 100 --max_eval_samples 50

        DDL running command:
        >>> python -m torch.distributed.launch --nproc_per_node number_of_gpu_you_have path_to_script.py
        
        Example:
        >>> CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 /home/chendo11/workfolder/TopTransformer/code_pkg/utils/topt_masked_pretrain.py

        # or select the specific GPUs and control their order
        >>> CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch /home/chendo11/workfolder/TopTransformer/code_pkg/utils/topt_masked_pretrain.py

        New version based on: https://pytorch.org/docs/stable/elastic/run.html
        >>> torchrun --nnodes=2 --nproc_per_node=2 /home/chendo11/workfolder/TopTransformer/code_pkg/utils/topt_masked_pretrain.py
    """
    TopT_Finetuning()


if __name__ == "__main__":
    main()
