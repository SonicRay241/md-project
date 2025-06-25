from model import Model, EncoderType
import pandas as pd

obesity_classifier = Model(
    "./data/ObesityDataSet2.csv", 
    "NObeyesdad",
    {'learning_rate': 0.02,
        'max_depth': 8,
        'n_estimators': 2000,
        'random_state': 69,
        'subsample': 0.8
    }
)

obesity_classifier.clean_data()
obesity_classifier.add_encoder(["Gender"], EncoderType.ONEHOT)
obesity_classifier.add_encoder(["MTRANS"], EncoderType.ONEHOT)
obesity_classifier.add_encoder(
    ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"],
    EncoderType.LABEL
)
obesity_classifier.add_encoder(
    ["CAEC", "CALC"],
    EncoderType.ORDINAL,
    categories=["no", "Sometimes", "Frequently", "Always"]
)
obesity_classifier.add_encoder(
    "target",
    EncoderType.ORDINAL,
    categories=["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
)

obesity_classifier.fit_transform_encoders()
obesity_classifier.train()

cols = obesity_classifier.df.columns.drop(labels="NObeyesdad")
data = ["Female",19,1.60,45.00,"no","no",3.00,3.00,"no","no",3.00,"yes",2.00,0.000,"no","Walking"]
test_df = pd.DataFrame(
    columns=cols,
    data=[data]
)
print(obesity_classifier.predict(test_df))

obesity_classifier.save("./model/model.pkl")