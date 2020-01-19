import os
import pandas as pd
import argparse

os.system('pip install joblib')
os.system('pip install imblearn')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Input variables
keep_vars = ['gender', 'age', 'income', 'day', 'member_from', 'dow',
             'n_transactions', 'avg_transctions',
             'n_offers_completed', 'n_offers_viewed', 'avg_reward',
             'reception_to_view_avg', 'view_to_completion_avg']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--val_split_ratio', type=float, default=0.2)
    parser.add_argument('--test_split_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1123)
    args = parser.parse_args()

    # This is the random seed, for reproducibility of results
    seed = args.seed
    # Check if target is correct
    tgt = args.target
    if tgt not in ['bogo', 'discount', 'info']:
        raise ValueError(
            ('Target argument must be "bogo", "discount" or "info" - '
             f'{tgt} passed')
        )

    # Read data
    input_path = os.path.join('/opt/ml/processing/input', f'{tgt}.csv')
    df = pd.read_csv(input_path, header=None, names=[tgt] + keep_vars)
    
    rus = RandomUnderSampler(random_state=seed)
    X_tot, y_tot = rus.fit_sample(df.drop(tgt, 1), df[tgt])

    # Split between train, validation and test data
    X, X_val, y, y_val = train_test_split(
        X_tot, y_tot, test_size=args.val_split_ratio,
        stratify=y_tot, random_state=seed
    )
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=args.test_split_ratio / (1 - args.val_split_ratio),
        stratify=y, random_state=seed
    )


    # Preprocessing pipeline for gender
    # Imputing with "O", then One Hot Encoding
    gender_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='constant', fill_value='O')),
            ('ohe', OneHotEncoder())
        ]
    )
    # Preprocessing pipeline for numeric features
    # Imputing with median, then standardizing distribution
    num_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    # Join the pipelines via a ColumnTransformer
    preprocessing = ColumnTransformer(transformers=[
        ('gender_pipeline', gender_pipe, ['gender']),
        ('numeric_pipeline', num_pipe, keep_vars[1:])
    ])

    # Fit the transformer
    X = preprocessing.fit_transform(X)
    X_val = preprocessing.transform(X_val)
    X_test = preprocessing.transform(X_test)

    # Save data
    pd.concat([y.reset_index(drop=True), pd.DataFrame(X)], axis=1).to_csv(
        os.path.join(f'/opt/ml/processing/output/{tgt}_train.csv'),
        header=False, index=False
    )
    pd.concat([y_val.reset_index(drop=True), pd.DataFrame(X_val)], axis=1).to_csv(
        os.path.join(f'/opt/ml/processing/output/{tgt}_val.csv'),
        header=False, index=False
    )
    pd.Series(y_test.reset_index(drop=True)).to_csv(
        os.path.join(f'/opt/ml/processing/output/{tgt}_test_tgt.csv'),
        header=False, index=False
    )
    pd.DataFrame(X_test).to_csv(
        os.path.join(f'/opt/ml/processing/output/{tgt}_test.csv'),
        header=False, index=False
    )

    # Save transformer
    joblib.dump(
        preprocessing, f'/opt/ml/processing/output/{tgt}_transformer.joblib'
    )
