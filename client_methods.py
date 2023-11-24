from __future__ import annotations

from typing import Union

import torch

from joint_ml._metric import Metric
from torch import IntTensor, nn
from torch.utils.data import Dataset, DataLoader

from src.data.get_dataset import get_walmart_dataset
from src.models.time_series_transformer import TimeSeriesTransformer
from src.utils.input_mask import get_src_trg_masks


def load_model(tabular_cat_features_size=2,
               tabular_cat_features_possible_nums=[45, 81],
               tabular_num_features_size=1,
               tabular_num_features_ffn_hidden_size=512,
               time_series_cat_features_size=1,
               time_series_cat_features_possible_nums=[2],
               time_series_numerical_features_size=5,
               enc_seq_len=24,
               dec_seq_len=2,
               batch_first=True,
               out_seq_len=1,
               dim_val=256,
               n_encoder_layers=3,
               n_decoder_layers=3,
               n_heads=8,
               dropout_encoder=0.2,
               dropout_decoder=0.2,
               dropout_pos_enc=0.1,
               dim_feedforward_encoder=512,
               dim_feedforward_decoder=512,
               num_predicted_features=1) -> torch.nn.Module:
    tabular_cat_features_possible_nums = IntTensor(tabular_cat_features_possible_nums)
    time_series_cat_features_possible_nums = IntTensor(time_series_cat_features_possible_nums)

    model = TimeSeriesTransformer(
        tabular_cat_features_size=tabular_cat_features_size,
        tabular_cat_features_possible_nums=tabular_cat_features_possible_nums,
        tabular_num_features_size=tabular_num_features_size,
        tabular_num_features_ffn_hidden_size=tabular_num_features_ffn_hidden_size,
        time_series_cat_features_size=time_series_cat_features_size,
        time_series_cat_features_possible_nums=time_series_cat_features_possible_nums,
        time_series_numerical_features_size=time_series_numerical_features_size,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        batch_first=batch_first,
        out_seq_len=out_seq_len,
        dim_val=dim_val,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        dropout_encoder=dropout_encoder,
        dropout_decoder=dropout_decoder,
        dropout_pos_enc=dropout_pos_enc,
        dim_feedforward_encoder=dim_feedforward_encoder,
        dim_feedforward_decoder=dim_feedforward_decoder,
        num_predicted_features=num_predicted_features
    )

    return model


def get_dataset(dataset_path: str, with_split: bool, test_ratio=0.2,
                val_ratio=0.2, enc_seq_len=24, dec_seq_len=2, target_seq_len=2) -> (Dataset, Dataset, Dataset):
    train_dataset, val_dataset, test_dataset = None, None, None
    if with_split:
        train_dataset, val_dataset, test_dataset = get_walmart_dataset(dataset_path, test_ratio=test_ratio,
                                                                       val_ratio=val_ratio, enc_seq_len=enc_seq_len,
                                                                       dec_seq_len=dec_seq_len,
                                                                       target_seq_len=target_seq_len)

    return train_dataset, val_dataset, test_dataset


def train(model: torch.nn.Module, train_set: torch.utils.data.Dataset, batch_size, enc_seq_len, output_sequence_length,
          lr, valid_set: torch.utils.data.Dataset = None) -> tuple[list[Metric], torch.nn.Module]:
    model.train()
    total_loss = 0.
    log_interval = 100

    src_mask, trg_mask = get_src_trg_masks(enc_seq_len=enc_seq_len, output_sequence_length=output_sequence_length)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    criterion = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_set, batch_size)

    num_batches = len(train_set) // batch_size

    mae_metric = Metric(name='MAE_train')

    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            for key in data.keys():
                data[key] = data[key].to('cuda:0')

        output = model(
            tabular_categorical_features=data['tabular_categorical_features'],
            tabular_numerical_features=data['tabular_numerical_features'],
            src_time_series_categorical_features=data['src_time_series_categorical_features'],
            src_time_series_numerical_features=data['src_time_series_numerical_features'],
            trg_time_series_categorical_features=data['trg_time_series_categorical_features'],
            trg_time_series_numerical_features=data['trg_time_series_numerical_features'],
            src_mask=src_mask,
            tgt_mask=trg_mask
        )

        loss = criterion(output, data['trg_y'])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            mae_metric.log_value(cur_loss)
            total_loss = 0

    return [mae_metric], model


def test(model: torch.nn.Module, test_set: torch.utils.data.Dataset, enc_seq_len, output_sequence_length) -> Union[
    list[Metric], tuple[list[Metric], list]]:
    model.eval()  # turn on evaluation mode
    total_loss = 0.

    test_dataloader = DataLoader(test_set, batch_size=1)
    criterion = nn.L1Loss(reduction='mean')

    src_mask, trg_mask = get_src_trg_masks(enc_seq_len=enc_seq_len, output_sequence_length=output_sequence_length)

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            if torch.cuda.is_available():
                for key in data.keys():
                    data[key] = data[key].to('cuda:0')

            output = model(
                tabular_categorical_features=data['tabular_categorical_features'],
                tabular_numerical_features=data['tabular_numerical_features'],
                src_time_series_categorical_features=data['src_time_series_categorical_features'],
                src_time_series_numerical_features=data['src_time_series_numerical_features'],
                trg_time_series_categorical_features=data['trg_time_series_categorical_features'],
                trg_time_series_numerical_features=data['trg_time_series_numerical_features'],
                src_mask=src_mask,
                tgt_mask=trg_mask
            )

            total_loss += criterion(output, data['trg_y']).item()

    mae_test_metric = Metric('MAE_test')
    mae_test_metric.log_value(total_loss / (len(test_dataloader)))

    return [mae_test_metric]


def get_prediction(model: torch.nn.Module, dataset_path: str, enc_seq_len, output_sequence_length) -> list:
    test_set = get_walmart_dataset(dataset_path, 0, 0, enc_seq_len, output_sequence_length, 0)

    test_dataloader = DataLoader(test_set, batch_size=1)

    src_mask, trg_mask = get_src_trg_masks(enc_seq_len=enc_seq_len, output_sequence_length=output_sequence_length)
    outputs = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            if torch.cuda.is_available():
                for key in data.keys():
                    data[key] = data[key].to('cuda:0')

            output = model(
                tabular_categorical_features=data['tabular_categorical_features'],
                tabular_numerical_features=data['tabular_numerical_features'],
                src_time_series_categorical_features=data['src_time_series_categorical_features'],
                src_time_series_numerical_features=data['src_time_series_numerical_features'],
                trg_time_series_categorical_features=data['trg_time_series_categorical_features'],
                trg_time_series_numerical_features=data['trg_time_series_numerical_features'],
                src_mask=src_mask,
                tgt_mask=trg_mask
            )
            outputs.append(output)
    return outputs
