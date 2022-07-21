# CPU Cinebench Predictor web app deployed on Heroku

The deployed web app is live at https://vssyguy-cpu.herokuapp.com/

This web app predicts the singleScore and multiScore of Cinebench R23 for an AMD or INTEL CPU as a function of their input parameters (manufacturer, cores, baseClock, turboClock, pc_type, tdp, date).

The web app was built in Python using the following libraries:

- streamlit
- pandas
- numpy
- scikit-learn
- lazypredict
- xgboost
- pickle
- matplotlib
- shap
