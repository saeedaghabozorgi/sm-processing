
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)



columns = ['sex', 'length', 'diameter', 'height', 'whole_weight',
           'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
class_labels = [' - 50000.', ' 50000+.']

def print_shape(df):
    negative_examples, positive_examples = np.bincount(df['income'])
    print('Data shape: {}, {} positive examples, {} negative examples'.format(df.shape, positive_examples, negative_examples))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))
    
    input_data_path = os.path.join('/opt/ml/processing/input', 'abalone_500m.csv')
    print(input_data_path)
    print('Reading input data from {}'.format(input_data_path))
    #df = pd.read_csv(input_data_path)
    df = pd.read_csv(input_data_path,header=None)
    df.columns = columns
    #df = df.drop('sex', axis=1)
    #df = pd.DataFrame(data=df, columns=columns)
    print('xxxxxx')
    print(df.head())
    
    split_ratio = args.train_test_split_ratio
    print('Splitting data into train and test sets with ratio {}'.format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(df.drop('rings', axis=1), df['rings'], test_size=split_ratio, random_state=0)
    preprocess = make_column_transformer(
        (['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight'], StandardScaler() ),
        (['sex'], OneHotEncoder(sparse=False)))
    print('Running preprocessing and feature engineering transformations')
    #print(X_train)
    train_features = preprocess.fit_transform(X_train)
    #print(train_features)
    test_features = preprocess.fit_transform(X_test)
    print('Train data shape after preprocessing: {}'.format(train_features.shape))
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
    
    
    print('Test data shape after preprocessing: {}'.format(test_features.shape))
    test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features.csv')
    
    
    print('Saving training features to {}'.format(train_features_output_path))
    #pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
    
    bigdf = pd.DataFrame(train_features)
    number_of_chunks = len(bigdf) // 1000000 + 1
    for id, df_i in  enumerate(np.array_split(bigdf, number_of_chunks)):
        print(str(id))
        train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features_{id}.csv'.format(id=id))
        pd.DataFrame(df_i).to_csv(train_features_output_path, header=False, index=False)

    
    
    print('Saving test features to {}'.format(test_features_output_path))
    #pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)
    bigdf = pd.DataFrame(test_features)
    number_of_chunks = len(bigdf) // 1000000 + 1
    for id, df_i in  enumerate(np.array_split(bigdf, number_of_chunks)):
        print(str(id))
        test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features_{id}.csv'.format(id=id))
        pd.DataFrame(df_i).to_csv(test_features_output_path, header=False, index=False)
        
    print('end')