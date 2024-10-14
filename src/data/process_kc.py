import argparse
import pandas as pd
import numpy as np
import os
import json


def process_kc(input_file, output_file=None, remove_duplicates=True, remove_missing=True):
    # Load the variables
    vars = json.load(open('config/data/kc.json'))
    # Load the data
    df = pd.read_csv(input_file, index_col=0)
    # Sort the data by date
    df = df.sort_values(vars['date_var'])
    # Log transform the price
    if 'log' not in vars['target']:
        df['log_price'] = np.log(df[vars['target']])
        price_col = 'log_price'

    if remove_missing:
        # Remove missing values based on all variables except yr_built and date
        df = df.dropna(subset=vars['hedonic_vars'][:-1]+vars['spatial_vars']+[price_col])
    if remove_duplicates:
        # Remove duplicates based on spatial variables
        df = df.drop_duplicates(subset=vars['spatial_vars'], keep='last')
    

    # time-based split
    df[vars['date_var']] = pd.to_datetime(df[vars['date_var']])
    df['set'] = pd.Series(dtype='object')
    df.loc[df[vars['date_var']] < pd.to_datetime("2014-12-08", format='%Y-%m-%d', yearfirst=True), 'set'] = 'train'
    df.loc[(df[vars['date_var']] >= pd.to_datetime("2014-12-08", format='%Y-%m-%d', yearfirst=True)) & \
                                (df[vars['date_var']] < pd.to_datetime("2015-03-23", format='%Y-%m-%d', yearfirst=True)), 'set']='val'
    df.loc[df[vars['date_var']] >= pd.to_datetime("2015-03-23", format='%Y-%m-%d', yearfirst=True), 'set'] = 'test'
    
    # Select the variables and rename some of them
    final_vars = vars['hedonic_vars'] + vars['spatial_vars'] + [vars['date_var']] + [price_col] + ['set']
    df = df[final_vars]
    df.rename(columns={vars['spatial_vars'][0]: 'x_geo', vars['spatial_vars'][1]:'y_geo', vars['date_var']:'transaction_date'}, inplace=True)
    
    # Save the data
    if output_file is not None:
        df.to_csv(output_file)
        print(f'File saved as {output_file}')
    return df



if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_duplicates', type=bool, default=True)
    parser.add_argument('--remove_missing', type=bool, default=True)
    parser.add_argument('--input_file', type=str, default='/data/raw/kc_final.csv')
    parser.add_argument('--output_file', type=str, default='/data/processed/kc.csv')

    args = parser.parse_args()

    df = process_kc(args.input_file, args.output_file, args.remove_duplicates, args.remove_missing)

