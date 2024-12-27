import pandas as pd
from pre_process import pre_process
from train_models import train_models


def classification(df : pd.DataFrame,
                    class_column : str,
                    path : str,
                    name : str) -> None:
    
    x_data , y = pre_process(df, class_column)

    train_models(x_data, y, path , name)




path_csv = "/home/reza/hamedan_seifi.csv"
df = pd.read_csv(path_csv)

def binary(lable, traget):
    if lable != traget:
        return "other"
    return lable

df["Name"] = df["Name"].apply(lambda x : binary(x, "wi-wr-br-bi"))

df = df.sample(100)
classification(df , "Name", "/home/reza/hamedan_seifi", "hamedan_seifi")