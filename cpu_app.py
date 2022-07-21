import streamlit as st
import pandas as pd
from models.building import Model

class App():
    def __init__(self):
        self.print_info()
        self.model = Model()
        self.gather_input()
        self.print_user_input()
        self.model.start(self.user_input_features)
        self.print_predictions()
        self.print_evaluations()
        self.print_feature_importance()
    
    def print_info(self):
        st.write("""
        # Performance prediction for AMD and INTEL upcoming CPUs

        This app predicts the **Cinebench R23 Score** of an upcoming CPU!!!
        
        Note that it will never be 100% accurate as there are many variables that are not taken into account.

        Data obtained from [kaggle](https://www.kaggle.com/datasets/alanjo/cpu-benchmarks) by Alan Jo.
        """)

        st.write('---')
    
    def gather_input(self):
        # Sidebar form
        st.sidebar.header('User Input Features')
        # Collect features from uploaded csv or form
        input_file = st.sidebar.file_uploader("Upload your features through a CSV file", type=['csv'])
        
        manufacturer =  st.sidebar.selectbox('Manufacturer', ('AMD', 'INTEL'))
        cpuName = st.sidebar.text_input('CPU Name', 'Ryzen 5 7600X')
        cores = st.sidebar.number_input('Number of Cores', 2, 256, 6, 2)
        baseClock = st.sidebar.slider('Frecuency (base)', 1.0, 7.0, 4.4, 0.1)
        turboClock = st.sidebar.slider('Frecuency (turbo)', 1.0, 7.0, 5.0, 0.1)
        pc_type = st.sidebar.selectbox('CPU Type/Category', ('Desktop','Laptop'))
        tdp = st.sidebar.number_input('TDP', 2, 400, 65, 1)
        date = st.sidebar.number_input('Release date (year)', 2000, 2100, 2022, 1)
        st.sidebar.markdown("\n")
        data = {'manufacturer': manufacturer,
                'cpuName': cpuName,
                'cores': cores,
                'baseClock': baseClock,
                'turboClock': turboClock,
                'type': pc_type,
                'TDP': tdp,
                'date': date
               }
        
        self.user_input_features = pd.read_csv(input_file) if input_file is not None else pd.DataFrame(data, index=[0])
    
    def print_user_input(self):
        # Print user specified parameters
        st.header('Specified Input Features')
        st.table(self.user_input_features)
        st.write('---')
        
    def highlight_prediction(self, s):
        return ['color: gray']*len(s) if s.name != self.user_input_features['cpuName'][0] else ['font-weight: bold']*len(s)
        
    def print_predictions(self):
        st.header('Predictions:')
        comparisons = self.model.compare(self.user_input_features['cpuName'][0])
        col1, col2 = st.columns([1,1])
        with col1:
            st.table(comparisons[0].style.apply(self.highlight_prediction, axis=1))
        with col2:
            st.table(comparisons[1].style.apply(self.highlight_prediction, axis=1))
    
    def print_evaluations(self):
        for target in self.model.targets:
            st.write(f'Evaluation for {target} predictor:')
            st.table(self.model.evaluations[target])
        
        st.write('---')

    def print_feature_importance(self):
        st.header('Feature Importance')
        self.model.shap_plot()

if __name__ == '__main__':
    App()
    

