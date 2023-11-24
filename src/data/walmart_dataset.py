import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Tuple


class WalmartDataset(Dataset):
    def __init__(self, data, enc_seq_len: int, dec_seq_len: int, target_seq_len: int):
        super().__init__()

        self.data = data  # (tabular_features,time_series_features)

        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        tabular_features, time_series_features = item
        src_time_series_categorical_features, src_time_series_numerical_features, trg_time_series_categorical_features, trg_time_series_numerical_features, trg_y = self.get_src_trg(
            time_series_features)

        tabular_categorical_features, tabular_numerical_features = tabular_features
        tabular_categorical_features, tabular_numerical_features = list(tabular_categorical_features), [
            tabular_numerical_features]

        return {
            'tabular_categorical_features': torch.IntTensor(tabular_categorical_features),
            'tabular_numerical_features': torch.Tensor(tabular_numerical_features),
            'src_time_series_categorical_features': torch.IntTensor(src_time_series_categorical_features),
            'src_time_series_numerical_features': torch.Tensor(src_time_series_numerical_features),
            'trg_time_series_categorical_features': torch.IntTensor(trg_time_series_categorical_features),
            'trg_time_series_numerical_features': torch.Tensor(trg_time_series_numerical_features),
            'trg_y': torch.Tensor(trg_y).reshape(self.target_seq_len, -1)
        }

    def get_src_trg(
            self,
            time_series_features
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # assert len(time_series_features) == self.enc_seq_len + self.target_seq_len, "Features length does not equal (input length + target length)"

        time_series_categorical_features = []
        time_series_numerical_features = []

        for t in range(len(time_series_features)):
            time_step = time_series_features[t]
            time_series_categorical_features.append(time_step[0])
            time_series_numerical_features.append(list(time_step[1]))

        src_time_series_categorical_features = time_series_categorical_features[:self.enc_seq_len]
        src_time_series_numerical_features = time_series_numerical_features[:self.enc_seq_len]

        trg_time_series_categorical_features = time_series_categorical_features[
                                               self.enc_seq_len - 1:self.enc_seq_len - 1 + self.dec_seq_len]
        trg_time_series_numerical_features = time_series_numerical_features[
                                             self.enc_seq_len - 1:self.enc_seq_len - 1 + self.dec_seq_len]

        # assert len(trg_time_series_numerical_features) == self.dec_seq_len, "Length of trg num does not match target sequence length"
        # assert len(trg_time_series_categorical_features) == self.dec_seq_len, "Length of trg cat does not match target sequence length"

        trg_y_numerical_features = time_series_numerical_features[-self.target_seq_len:]

        trg_y = [data_point[0] for data_point in trg_y_numerical_features]

        return src_time_series_categorical_features, src_time_series_numerical_features, trg_time_series_categorical_features, trg_time_series_numerical_features, trg_y


class StoreSales:
    def __init__(self, store, dept, size, week, weekly_sales, is_holiday, temperature, cpi, unemployment,
                 indices_window_size, indices_step_size):
        self.store = store
        self.dept = dept
        self.size = size

        self.week = week
        self.week_sin = self.sin_transform(self.week)
        self.week_cos = self.cos_transform(self.week)

        self.weekly_sales = weekly_sales
        self.is_holiday = is_holiday
        self.temperature = temperature
        self.cpi = cpi
        self.unemployment = unemployment

        self.indices = self.get_indices(window_size=indices_window_size, step_size=indices_step_size)

    def get_indices(self, window_size, step_size):
        stop_position = len(self.weekly_sales) - 1

        subseq_first_idx = 0

        subseq_last_idx = window_size

        indices = []

        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx))

            subseq_first_idx += step_size

            subseq_last_idx += step_size

        return indices

    def get_sequence(self, index):
        tabular_categorical_features = (self.store, self.dept)
        tabular_numerical_features = (self.size)
        tabular_features = [tabular_categorical_features, tabular_numerical_features]

        start_idx = self.indices[index][0]
        end_idx = self.indices[index][1]

        weeks = self.week[start_idx:end_idx]
        weeks_sin = self.week_sin[start_idx:end_idx]
        weeks_cos = self.week_cos[start_idx:end_idx]

        time_series = self.weekly_sales[start_idx:end_idx]

        holidays = self.is_holiday[start_idx:end_idx]
        temperatures = self.temperature[start_idx:end_idx]
        cpis = self.cpi[start_idx:end_idx]
        unemployments = self.unemployment[start_idx:end_idx]

        time_series_features = [
            ((holidays[i]), (time_series[i], weeks_sin[i], weeks_cos[i], temperatures[i], cpis[i], unemployments[i]))
            for i in range(len(time_series))]

        return (tabular_features, time_series_features)

    def sin_transform(self, values):
        values = np.array(values)
        return np.sin(2 * np.pi * values / 52)

    def cos_transform(self, values):
        values = np.array(values)
        return np.cos(2 * np.pi * values / 52)


def get_temp_store_sales(df, indices_window_size=26, indices_step_size=1):
    stores_sales = []
    store_nums = sorted(list(df.Store.unique()))
    for store_num in store_nums:
        df_store = df[df.Store == store_num]

        dept_nums = sorted(list(df_store.Dept.unique()))

        for dept_num in dept_nums:
            df_store_dept = df_store[df_store.Dept == dept_num]

            store_dept_sales = StoreSales(
                store=store_num,
                dept=dept_num,
                size=float(df_store_dept.Size.unique()[0]),
                week=df_store_dept.Week.tolist(),
                weekly_sales=df_store_dept.Weekly_Sales.tolist(),
                is_holiday=df_store_dept.IsHoliday.tolist(),
                temperature=df_store_dept.Temperature.tolist(),
                cpi=df_store_dept.CPI.tolist(),
                unemployment=df_store_dept.Unemployment.tolist(),
                indices_window_size=indices_window_size,
                indices_step_size=indices_step_size)

            stores_sales.append(store_dept_sales)
    return stores_sales

def get_subsequences(store_sales):
    subsequences = []
    for store_sale in store_sales:
        for subsequence_i in range(len(store_sale.indices)):
            subsequent = store_sale.get_sequence(subsequence_i)
            subsequences.append(subsequent)
    return subsequences
