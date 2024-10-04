import pandas as pd

from typing import List


class Dataset:
    data: pd.DataFrame

    unit_strs: List[str] = ["mcg", "mg", "g", "IU"]
    macro_columns: List[str] = [
        "serving_size",
        "calories",
        "total_fat",
        "saturated_fat",
        "protein",
        "carbohydrate",
        "fiber",
        "sugars",
        "fructose",
        "galactose",
        "glucose",
        "lactose",
        "maltose",
        "sucrose",
        "saturated_fatty_acids",
        "monounsaturated_fatty_acids",
        "polyunsaturated_fatty_acids",
        "water",
    ]

    def read_process_dataset(self, filepath: str):
        self.read_dataset(filepath)
        self.data = self.clean_values(self.data)
        self.data = self.process_columns(self.data)

    def read_dataset(self, filepath: str):
        self.data = pd.read_excel(filepath, index_col=0)

    def clean_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda x: [self.str2float(v) for v in x])

    def str2float(self, value: str) -> float:
        if not isinstance(value, str):
            return value
        try:
            for u in [" ", *self.unit_strs]:
                value = value.replace(u, "")
            return float(value)
        except:
            return value

    def process_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        # Name
        df_name = df.name
        df_name.name = ("identifier", "name")

        # Tags
        df_tags = pd.DataFrame(df.name.str.split(",").tolist()).dropna()
        df_tags.columns = pd.MultiIndex.from_product([["identifier"], df_tags.columns])
        df_tags.insert(0, ("identifier", "name"), 9)
        df_tags.loc[:, pd.IndexSlice["identifier", "name"]] = df.name

        # Identify columns as macro or micronutrients
        df_values = df.loc[:, [c != "name" for c in df.columns]]
        df_macros = df_values.loc[
            :, [c in self.macro_columns for c in df_values.columns]
        ]
        df_macros.columns = pd.MultiIndex.from_product([["macro"], df_macros.columns])
        df_micros = df_values.loc[
            :, [c not in self.macro_columns for c in df_values.columns]
        ]
        df_micros.columns = pd.MultiIndex.from_product([["micro"], df_micros.columns])
        return pd.concat([df_name, df_tags, df_macros, df_micros])
