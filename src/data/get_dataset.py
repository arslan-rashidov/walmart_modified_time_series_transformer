import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, QuantileTransformer
from src.data.walmart_dataset import get_temp_store_sales, get_subsequences, WalmartDataset


def get_walmart_dataset(merged_input_filepath, test_ratio=0.2,
                        val_ratio=0.2, enc_seq_len=24, dec_seq_len=2, target_seq_len=2):
    X_train, X_val, X_test = preprocess_dataset(merged_input_filepath, test_ratio, val_ratio)

    train_store_sales = get_temp_store_sales(X_train)
    val_store_sales = get_temp_store_sales(X_val)
    test_store_sales = get_temp_store_sales(X_test)

    train_sequences = get_subsequences(train_store_sales)
    val_sequences = get_subsequences(val_store_sales)
    test_sequences = get_subsequences(test_store_sales)

    train_dataset = WalmartDataset(
        data=train_sequences,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=target_seq_len
    )

    val_dataset = WalmartDataset(
        data=val_sequences,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=target_seq_len
    )

    test_dataset = WalmartDataset(
        data=test_sequences,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=target_seq_len
    )

    return train_dataset, val_dataset, test_dataset


def preprocess_dataset(merged_input_filepath, test_ratio, val_ratio):
    X = pd.read_csv(merged_input_filepath)

    X = X.drop(columns=["MarkDown1", 'MarkDown2', 'MarkDown3', "MarkDown4", 'MarkDown5', 'Type', 'Fuel_Price'], axis=1)
    X.Date = pd.to_datetime(X.Date)
    X = X.sort_values(by=['Date', 'Store'])
    X['Week'] = X.Date.apply(lambda x: x.isocalendar()[1])
    X = X.drop(['Date'], axis=1)

    #### Splitting

    X_len = X.shape[0]
    test_size = int(X_len * test_ratio)
    val_size = int(X_len * val_ratio)

    X_test = X.iloc[X_len - test_size:].copy()
    X_val = X.iloc[X_len - test_size - val_size:X_len - test_size].copy()
    X_train = X.iloc[:X_len - test_size - val_size].copy()

    #### Preprocessing Store

    store_label_encoder = LabelEncoder()
    store_label_encoder.fit(sorted(list(X_train.Store.unique())))

    X_train['Store'] = store_label_encoder.transform(X_train[['Store']])
    X_val['Store'] = store_label_encoder.transform(X_val[['Store']])
    X_test['Store'] = store_label_encoder.transform(X_test[['Store']])

    #### Preprocessing Dept

    dept_label_encoder = LabelEncoder()
    dept_label_encoder.fit(sorted(list(X_train.Dept.unique())))

    X_train['Dept'] = dept_label_encoder.transform(X_train[['Dept']])
    X_val['Dept'] = dept_label_encoder.transform(X_val[['Dept']])
    X_test['Dept'] = dept_label_encoder.transform(X_test[['Dept']])

    #### Preprocessing Size

    size_min_max_scaler = MinMaxScaler()
    size_min_max_scaler.fit(X_train[['Size']])

    X_train.Size = size_min_max_scaler.transform(X_train[['Size']])
    X_val.Size = size_min_max_scaler.transform(X_val[['Size']])
    X_test.Size = size_min_max_scaler.transform(X_test[['Size']])

    #### Preprocessing IsHoliday

    is_holiday_transformer = lambda x: 1 if x else 0

    X_train.IsHoliday = X_train.IsHoliday.apply(is_holiday_transformer)
    X_val.IsHoliday = X_val.IsHoliday.apply(is_holiday_transformer)
    X_test.IsHoliday = X_test.IsHoliday.apply(is_holiday_transformer)

    #### Preprocessing Temperature

    temperature_min_max_scaler = MinMaxScaler()
    temperature_min_max_scaler.fit(X_train[['Temperature']])

    X_train.Temperature = temperature_min_max_scaler.transform(X_train[['Temperature']])
    X_val.Temperature = temperature_min_max_scaler.transform(X_val[['Temperature']])
    X_test.Temperature = temperature_min_max_scaler.transform(X_test[['Temperature']])

    #### Preprocessing CPI

    cpi_min_max_scaler = MinMaxScaler()
    cpi_min_max_scaler.fit(X_train[['CPI']])

    X_train.CPI = cpi_min_max_scaler.transform(X_train[['CPI']])
    X_val.CPI = cpi_min_max_scaler.transform(X_val[['CPI']])
    X_test.CPI = cpi_min_max_scaler.transform(X_test[['CPI']])

    #### Preprocessing Unemployment

    unemployment_min_max_scaler = MinMaxScaler()
    unemployment_min_max_scaler.fit(X_train[['Unemployment']])

    X_train.Unemployment = unemployment_min_max_scaler.transform(X_train[['Unemployment']])
    X_val.Unemployment = unemployment_min_max_scaler.transform(X_val[['Unemployment']])
    X_test.Unemployment = unemployment_min_max_scaler.transform(X_test[['Unemployment']])

    #### Preprocessing Weekly_Sales

    weekly_sales_quantile_transformer = QuantileTransformer()
    weekly_sales_quantile_transformer.fit(X_train[['Weekly_Sales']])

    X_train.Weekly_Sales = weekly_sales_quantile_transformer.transform(X_train[['Weekly_Sales']])
    X_val.Weekly_Sales = weekly_sales_quantile_transformer.transform(X_val[['Weekly_Sales']])
    X_test.Weekly_Sales = weekly_sales_quantile_transformer.transform(X_test[['Weekly_Sales']])

    return X_train, X_val, X_test
