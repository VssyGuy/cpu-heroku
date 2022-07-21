import pandas as pd

class Data():
    def __init__(self):
        self.merge()

    def merge(self):
        # Resource from https://www.kaggle.com/datasets/alanjo/cpu-benchmarks
        passmark = pd.read_csv('data/CPU_benchmark_v4.csv')
        cinebench = pd.read_csv('data/CPU_r23_v2.csv')
        
        # Match data names
        passmark['cpuName'].replace('-', ' ', regex=True, inplace=True)
        passmark['cpuName'].replace('Ryzen Threadripper', 'Threadripper', regex=True, inplace=True)
        passmark['cpuName'] = passmark['cpuName'].str.lower()
        cinebench['cpuName'] = cinebench['cpuName'].str.lower()
        
        # Fix cpuName not being explicit
        passmark['cpuName'] = passmark['cpuName'].str.extract(r'[a-zA-Z]+([^@]+)')
        
        # Remove trailing spaces
        passmark['cpuName'] = passmark['cpuName'].str.rsplit().str.join(' ')
        cinebench['cpuName'] = cinebench['cpuName'].str.rsplit().str.join(' ')
        self.df = pd.merge(cinebench, passmark[['cpuName', 'TDP', 'testDate']], on='cpuName', how="inner")
        self.df.dropna(how='any', inplace=True)
        self.df.rename(columns = {'testDate':'date'}, inplace = True)
        