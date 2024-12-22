import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
import lightgbm as lgb

class OrdinalClassifier(BaseEstimator):
    def __init__(self, clfs, clfs_args, reverse_classes = False):
        self.clfs = clfs
        self.clfs_args = clfs_args
        self.reverse_classes = reverse_classes
        self.classes = None

    def fit_eval(self, X, y, X_val, y_val, early_stopping_round, cat_cols):
        self.classes = np.sort(np.unique(y))
        assert len(self.classes) >= 3, 'OrdinalClassifier needs at least 3 classes.'

        if self.reverse_classes:
            self.classes = self.classes[::-1]
        
        self.clf_ords = []
        for i, threshold in enumerate(self.classes[:-1]):
            y_binary = (y > threshold).astype(int) if not self.reverse_classes else (y < threshold).astype(int)
            y_val_binary = (y_val > threshold).astype(int) if not self.reverse_classes else (y_val < threshold).astype(int)

            clf_models = []
            for j, clf in enumerate(self.clfs):
                clf_args = self.clfs_args[j]

                clf_name = clf.__name__ if isinstance(clf, type) else clf.__class__.__name__
                if 'LGB' in clf_name or 'LightGBM' in clf_name:
                    classifier = clf(**clf_args)
                    classifier.fit(X, y_binary, eval_set = [(X_val, y_val_binary)], callbacks=[lgb.early_stopping(early_stopping_round, verbose = False)], categorical_feature = cat_cols)

                elif 'CAT' in clf_name or 'CatBoost' in clf_name:
                    train_data = Pool(data = X, label = y_binary, cat_features = cat_cols)
                    valid_data = Pool(data = X_val, label = y_val_binary, cat_features = cat_cols)

                    classifier = clf(**clf_args)
                    classifier.fit(train_data, eval_set = valid_data, verbose = False)
                  
                else:
                    classifier = clf(**clf_args)
                    classifier.fit(X, y_binary, eval_set=[(X_val, y_val_binary)], verbose = False)
                
                clf_models.append(classifier)
              
            self.clf_ords.append(clf_models)

    def fit(self, X, y, cat_cols):
        self.classes = np.sort(np.unique(y))
        assert len(self.classes) >= 3, 'OrdinalClassifier needs at least 3 classes.'

        if self.reverse_classes:
            self.classes = self.classes[::-1]

        self.clf_ords = []
        for i, threshold in enumerate(self.classes[:-1]):
            y_binary = (y > threshold).astype(int) if not self.reverse_classes else (y < threshold).astype(int)
            
            clf_models = []
            for j, clf in enumerate(self.clfs):
                clf_args = self.clfs_args[j]

                clf_name = clf.__name__ if isinstance(clf, type) else clf.__class__.__name__
                if 'LGB' in clf_name or 'LightGBM' in clf_name:
                    classifier = clf(**clf_args)
                    classifier.fit(X, y_binary, categorical_feature = cat_cols)

                elif 'CAT' in clf_name or 'CatBoost' in clf_name: 
                    classifier = clf(**clf_args)
                    classifier.fit(X, y_binary, verbose = False)

                else:
                    classifier = clf(**clf_args)
                    classifier.fit(X, y_binary)
              
                clf_models.append(classifier)
                
            self.clf_ords.append(clf_models)

    def predict_proba(self, X, conditional = True):
        probas = []

        for clf_ord in self.clf_ords:
            class_probas = []
            for clf_mls in clf_ord:
                class_probas.append(clf_mls.predict_proba(X)[:, 1])
            probas.append(np.mean(class_probas, axis = 0))
        
        ensemble_probas = np.array(probas)

        p = np.zeros((X.shape[0], len(self.classes)))

        if self.reverse_classes:
            if conditional:
                p[:, -1] = ensemble_probas[-1]
                for k in range(len(ensemble_probas) - 2, -1, -1):
                    p[:, k] = ensemble_probas[k] * p[:, k + 1]
                p[:, 0] = 1
            
            else:
                p[:, -1] = ensemble_probas[-1]
                for k in range(len(ensemble_probas) - 2, -1, -1):
                    p[:, k] = p[:, k + 1] - ensemble_probas[k]
                p[:, 0] = 1 - p[:, 0]
                
        else:
            if conditional:
                p[:, 0] = 1 - ensemble_probas[0]
                for k in range(1, len(self.classes) - 1):
                    p[:, k] = ensemble_probas[k - 1] * (1 - ensemble_probas[k])
                p[:, -1] = ensemble_probas[-1]

            else:
                p[:, 0] = 1 - ensemble_probas[0]
                for k in range(1, len(self.classes) - 1):
                    p[:, k] = ensemble_probas[k - 1] - ensemble_probas[k]
                p[:, -1] = ensemble_probas[-1]
          
        #p /= p.sum(axis=1, keepdims=True)
        p_sum = p.sum(axis=1, keepdims=True)
        p_sum[p_sum == 0] = 1
        p /= p_sum
        return p
        
    def predict(self, X, conditional = True):
        probas = self.predict_proba(X, conditional = conditional)

        if self.reverse_classes:
            probas_argmax = np.argmin(probas, axis=1)

        else:
            probas_argmax = np.argmax(probas, axis=1)

        return probas_argmax

    def get_feature_importances(self, feature_names):
        feature_importances = []

        for clf_ord in self.clf_ords:
            for clf in clf_ord:
                clf_name = clf.__class__.__name__

                if 'XGB' in clf_name:
                    importances = clf.get_booster().get_score() # importance_type='gain'
                elif 'LGB' in clf_name or 'LightGBM' in clf_name:
                    importances = clf.booster_.feature_importance() # importance_type='gain'
                elif 'CAT' in clf_name or 'CatBoost' in clf_name:
                    importances = clf.get_feature_importance(type='PredictionValuesChange')
                else:
                    importances = clf.feature_importances_

                if isinstance(importances, dict):
                    importances = [importances.get(name, 0) for name in feature_names]

                feature_importances.append(importances)

        if feature_importances:
            feature_importances_avg = np.mean(feature_importances, axis=0)
            return feature_importances_avg
        else:
            raise ValueError('No feature importances')

    def shap_values(self, X):
        shap_values = []

        for clf_ord in self.clf_ords:
            for clf in clf_ord:
                explainer = shap.TreeExplainer(clf)
                shap_value = explainer.shap_values(X)
                shap_values.append(shap_value)

        if shap_values:
            shap_values_avg = np.mean(np.abs(np.array(shap_values)), axis = 0)
            return shap_values_avg
        else:
            raise ValueError('No shap values')