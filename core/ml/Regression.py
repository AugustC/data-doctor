from core.ml.BaseMLModels import BaseMLModels
import xgboost as xgb

class Regression(BaseMLModels):
    def __init__(self, params=None):
        super().__init__()
        if params is None:  # Default parameters for the XGBoost model
            params = {
                'n_estimators' : 100,
                'max_depth': 6,
                'learning_rate': 0.3,
                'objective': 'reg:squarederror',
                'enable_categorical': True,
            }
        self.model = xgb.XGBRegressor(**params)
