import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Preprocess:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = None
        self.read_csv()
        self.fill_missing()

    def read_csv(self):
        self.df = pd.DataFrame()
        with open(self.file_name, 'r') as f:
            for line in f:
                self.df = pd.concat([self.df, pd.DataFrame([tuple(line.strip().split(','))])], ignore_index=True)
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        # self.reverse_df()

    def get_dataframe(self):
        return self.df

    def reverse_df(self):
        self.df = self.df.iloc[:, ::-1]
        self.df = self.df[::-1]
        self.df.reset_index(inplace=True, drop=True)
        self.df.columns = [x for x in range(0, self.df.shape[1])]

    def plot_first_row(self):
        plt.plot(self.df.iloc[1, :], [x for x in range(self.df.shape[1])])
        plt.show()

    def fill_missing(self):
        self.df = self.df.dropna(thresh=(3/4)*self.df.shape[1])
        self.df.reset_index(drop=True, inplace=True)
        # self.df = self.df.interpolate(method="linear", limit_direction='backward', axis=0)
        self.df = self.df.bfill(axis=1)
        # print(self.df)