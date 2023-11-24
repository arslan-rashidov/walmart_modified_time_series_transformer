# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.get_dataset import preprocess_dataset


@click.command()
@click.argument('train_sales_input_filepath', type=click.Path(exists=True))
@click.argument('features_input_filepath', type=click.Path(exists=True))
@click.argument('stores_input_filepath', type=click.Path(exists=True))
@click.argument('test_ratio', type=click.Parameter())
@click.argument('val_ratio', type=click.Parameter())
@click.argument('output_filepath', type=click.Path())
def main(train_sales_input_filepath, features_input_filepath, stores_input_filepath, test_ratio, val_ratio,
         output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    X_train, X_val, X_test = preprocess_dataset(train_sales_input_filepath, features_input_filepath,
                                                stores_input_filepath, test_ratio, val_ratio)

    X_train.to_csv(output_filepath + '/train.csv', index=False)
    X_val.to_csv(output_filepath + '/val.csv', index=False)
    X_test.to_csv(output_filepath + '/test.csv', index=False)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
