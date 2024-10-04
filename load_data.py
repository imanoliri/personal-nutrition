from dataset import Dataset

dataset_filepath = "data/nutrition.xlsx"
dataset_filepath_save = "data/nutrition.csv"
ds = Dataset()
ds.read_process_dataset(dataset_filepath)
ds.data.to_csv(dataset_filepath_save)
