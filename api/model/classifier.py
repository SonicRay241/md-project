import pandas as pd
from enum import Enum
from typing import List, Literal, Any
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier
import joblib

class EncoderType(Enum):
    LABEL = "label"
    ORDINAL = "ordinal"
    ONEHOT = "onehot"

class Model:
    def __init__(self, dataset_path: str, target_column: str, params: Any = {}):
        if not isinstance(dataset_path, str):
            raise TypeError("dataset_path must be type string.")
        
        if not isinstance(target_column, str):
            raise TypeError("target_column must be type string.")

        self.df: pd.DataFrame = pd.read_csv(dataset_path)
        self.target_column: str = target_column

        self.encoded_df: pd.DataFrame | None = None
        self.classifier: XGBClassifier = XGBClassifier(**params)
        self.encoders = {}

    def clean_data(self):
        # Dropping NULL and duplicates
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)

        # Extract digit only and chante type to numeric
        self.df['Age'] = self.df['Age'].astype(str).str.extract(r'(\d+)')
        self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce')
        self.df.reset_index(drop=True, inplace=True)

    def add_encoder(
            self,
            columns: List[str] | Literal["target"],
            encoder: EncoderType,
            categories: List[str] | None = None
        ):
        if not isinstance(columns, str) and (not isinstance(columns, list) or any(not isinstance(col, str) for col in columns)):
            raise TypeError("columns must be of type List[str] or Literal[\"target\"].")

        if columns == []:
            raise ValueError("columns should not be empty.")

        if isinstance(columns, list) and not set(columns).issubset(self.df.columns) and not "target":
            raise ValueError("columns list has one or more of unavailable columns from the data.")
        
        if not isinstance(encoder, EncoderType):
            raise TypeError("encoder must be of type EncoderType.")
        
        if encoder is EncoderType.ORDINAL: 
            if categories is None:
                raise AttributeError("categories list must be provided for Ordinal encoding.")
            
            if not isinstance(categories, list) or any(not isinstance(col, str) for col in categories):
                raise TypeError("categories must be of type List[str].")
        
        def get_encoder():
            match encoder:
                case EncoderType.LABEL:
                    return LabelEncoder()
                
                case EncoderType.ORDINAL:
                    return OrdinalEncoder(categories=[categories]).set_output(transform="pandas")
                
                case EncoderType.ONEHOT:
                    return OneHotEncoder(sparse_output=False).set_output(transform="pandas")

                case _: raise NotImplementedError("what?")

        if columns == "target":
            self.encoders["target"] = get_encoder()
            return

        for column in columns:
            self.encoders[column] = get_encoder()
    
    def fit_transform_encoders(self):
        encoder_keys = list(self.encoders.keys())
        categorical_columns = [self.target_column if col == "target" else col for col in encoder_keys]
        new_df = self.df.drop(columns=categorical_columns)

        for k in encoder_keys:
            colname = self.target_column if k == "target" else k
            match self.encoders[k].__class__.__name__:
                case "LabelEncoder":
                    encoded_col = pd.DataFrame(
                        self.encoders[k].fit_transform(self.df[[colname]]),
                        columns=[colname],
                        index=self.df.index
                        )
                
                case _:
                    encoded_col = self.encoders[k].fit_transform(self.df[[colname]])

            new_df = pd.concat([
                new_df,
                encoded_col
            ], axis=1)

        self.encoded_df = new_df
        return self.encoded_df
    
    def train(self):
        self.classifier.fit(
            self.encoded_df.drop(columns=self.target_column),
            self.encoded_df[self.target_column]
        )

    def predict(self, values: pd.DataFrame):
        if list(values.columns) != list(self.df.drop(columns=[self.target_column]).columns):
            raise TypeError(f"Predictor must have these columns: {self.df.columns.drop([self.target_column])}")
        
        encoder_keys = list(self.encoders.keys())
        encoder_keys.remove("target")
        pred_df = values.drop(columns=encoder_keys)

        for k in encoder_keys:
            match self.encoders[k].__class__.__name__:
                case "LabelEncoder":
                    encoded_col = pd.DataFrame(
                        self.encoders[k].transform(values[[k]]),
                        columns=[k],
                        index=pred_df.index
                        )
                case _:
                    encoded_col = self.encoders[k].transform(values[[k]])

            pred_df = pd.concat([
                pred_df,
                encoded_col
            ], axis=1)
        
        return self.encoders["target"].inverse_transform(self.classifier.predict(pred_df).reshape(-1, 1))[0]
        

    @classmethod
    def load(cls, path: str):
        instance = joblib.load(path)

        if not isinstance(instance, cls):
            raise TypeError(
                f"The object loaded must be an instance of {cls.__name__}, but got an instance of {type(instance).__name__}"
            )
        
        return instance
    
    def save(self, path: str):
        joblib.dump(self, path)