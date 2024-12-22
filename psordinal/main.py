import argparse
import json
from .module import OrdinalClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier

def main():
    parser = argparse.ArgumentParser(description = 'Run OrdinalClassifier with specified parameters.')

    parser.add_argument('--clfs', nargs='+', required=True, help='List of classifiers (e.g., XGBClassifier, CatboostClassifier, LGBMClassifier)')
    parser.add_argument('--clfs_args', nargs='+', required=True, help='List of JSON for classifier arguments')
    parser.add_argument('--reverse_classes', choices=[True, False], type=bool, required=False, default=False, help='Reverse the order of classes')
    
    parser.add_argument('--fit', action='store_true', help='Falg to train the model')
    parser.add_argument('--predict', action='store_true', help='Flag to run prediction')
    parser.add_argument('--test_data', type=str, required=False, help='Test data for prediction (as JSON string)')
    parser.add_argument('--train_data', type=str, required=False, help='Train data (as JSON string)')
    parser.add_argument('--train_labels', type=str, required=False, help='Train labels (as JSON string)')
    parser.add_argument('--cat_cols', type=str, required=False, help='Argument for categorical variables')

    args = parser.parse_args()

    try:
        clfs_args = [json.loads(arg) for arg in args.clfs_args]
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON format for --clfs_args: {args.clfs_args}') from e

    clfs = [eval(clf) for clf in args.clfs]

    clfs_ = []
    for clf_name in args.clfs:
      clfs_.append(clf_name)
    
    print(f'Classifiers: {clfs_}')
    print(f'Classifiers Arguments: {clfs_args}')

    ordinal_clf = OrdinalClassifier(
      clfs=clfs,
      clfs_args=clfs_args,
      reverse_classes=args.reverse_classes
    )

    if args.fit:
      if not args.train_data or not args.train_labels:
        raise ValueError('Train data and labels are required for traning')    

      train_data = np.array(json.loads(args.train_data))
      train_labels = np.array(json.loads(args.train_labels))

      ordinal_clf.fit(train_data, train_labels, args.cat_cols)
    
    else:
      raise ValueError('You must specify --fit')

    if args.predict:
      if not args.test_data:
        raise ValueError('Test data for prediction is required')

      test_data = np.array(json.loads(args.test_data))
      predictions = ordinal_clf.predict(test_data)
      print('Predictions:', predictions)
    
    else:
      raise ValueError('You must specify --predict')

if __name__ == "__main__":
    main()