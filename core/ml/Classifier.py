from core.ml.BaseMLModels import BaseMLModels
import xgboost as xgb

class Classifier(BaseMLModels):
    def __init__(self, params=None):
        super().__init__()
        if params is None:  # Default parameters for the XGBoost model
            params = {
                'n_estimators' : 100,
                'max_depth': 3,
                'learning_rate': 0.01,
                'objective': 'multi:softmax',
                'subsample': 0.8,
                'colsample_bytree' : 0.8,
                'gamma': 0.1,
                'num_class': 4,
                'eval_metric': ['merror', 'mlogloss'],
                'enable_categorical': True,
            }
        self.model = xgb.XGBClassifier(**params)
