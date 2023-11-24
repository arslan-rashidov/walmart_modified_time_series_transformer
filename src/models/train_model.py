import time
from pathlib import Path

import torch
from torch import nn, IntTensor
from torch.utils.data import DataLoader

from src.data.get_dataset import get_walmart_dataset
from src.models.time_series_transformer import TimeSeriesTransformer
from src.utils.input_mask import get_src_trg_masks


def fit(train_dataset, val_dataset, test_dataset):
    model = TimeSeriesTransformer(
        tabular_cat_features_size=2,
        tabular_cat_features_possible_nums=IntTensor([45, 81]),
        tabular_num_features_size=1,
        tabular_num_features_ffn_hidden_size=512,
        time_series_cat_features_size=1,
        time_series_cat_features_possible_nums=IntTensor([2]),
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
        num_predicted_features=1
    )

    src_mask, trg_mask = get_src_trg_masks(enc_seq_len=24, output_sequence_length=2)

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.L1Loss(reduction='mean')
    lr = 0.0001  # learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train(model: nn.Module, epoch: int) -> None:
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 100
        start_time = time.time()

        num_batches = len(train_dataset) // batch_size
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
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                print(f'| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {cur_loss}')
                total_loss = 0
                start_time = time.time()

    def evaluate(model: nn.Module, eval_dataloader) -> float:
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for i, data in enumerate(eval_dataloader):
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
        return total_loss / (len(eval_dataloader))

    print(f"VAL LOSS: {evaluate(model, val_dataloader)}")
    print(f"TEST LOSS: {evaluate(model, test_dataloader)}")
    for i in range(0, 300):
        train(model, i)
        print(f"VAL LOSS: {evaluate(model, val_dataloader)}")
        print(f"TEST LOSS: {evaluate(model, test_dataloader)}")
        torch.save(model.state_dict(), f"../models/model_improved_{i + 1}.pth")

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_walmart_dataset('/Users/astonuser/PycharmProjects/walmart_modified_time_series_transformer/data/raw/train.csv', '/Users/astonuser/PycharmProjects/walmart_modified_time_series_transformer/data/raw/features.csv', '/Users/astonuser/PycharmProjects/walmart_modified_time_series_transformer/data/raw/stores.csv')
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    fit(train_dataset, val_dataset, test_dataset)

