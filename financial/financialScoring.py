import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.externals import joblib 
import warnings
warnings.filterwarnings('ignore')

def financialScore(user_data):
    test2 = pd.read_csv('financial/train.csv')
    test2 = test2[:1799]
    test2 = test2.rename(columns={'Unnamed: 0': 'Unknown',
                                  'SeriousDlqin2yrs': 'Target',
                                  'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
                                  'NumberOfTime30-59DaysPastDueNotWorse': 'Late3059',
                                  'DebtRatio': 'DebtRatio',
                                  'MonthlyIncome': 'MonthlyIncome',
                                  'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
                                  'NumberOfTimes90DaysLate': 'Late90',
                                  'NumberRealEstateLoansOrLines': 'PropLines',
                                  'NumberOfTime60-89DaysPastDueNotWorse': 'Late6089',
                                  'NumberOfDependents': 'Deps'})
    test2.drop(labels=["Unknown"],axis = 1,inplace=True)
    test2.drop(labels=["Target"],axis = 1,inplace=True)

    test = pd.DataFrame(user_data, index=[0])

    ftest = pd.concat(objs=[test,test2], axis=0).reset_index(drop=True)

    ftest.MonthlyIncome = ftest.MonthlyIncome.fillna(ftest.MonthlyIncome.median())
    ftest.age = ftest.age.fillna(ftest.age.median())
    ftest.UnsecLines = ftest.UnsecLines.fillna(ftest.UnsecLines.median())
    ftest.Late3059 = ftest.Late3059.fillna(ftest.Late3059.median())
    ftest.DebtRatio = ftest.DebtRatio.fillna(ftest.DebtRatio.median())
    ftest.OpenCredit = ftest.OpenCredit.fillna(ftest.OpenCredit.median())
    ftest.PropLines = ftest.PropLines.fillna(ftest.PropLines.median())
    ftest.Late6089 = ftest.Late6089.fillna(ftest.Late6089.median())
    ftest.Late90 = ftest.Late90.fillna(ftest.Late90.median())
    ftest.Deps = ftest.Deps.fillna(ftest.Deps.median())

    for i in range(len(ftest)):
        if ftest.Late3059[i] >= 6:
            ftest.Late3059[i] = 6
    for i in range(len(ftest)):
        if ftest.Late90[i] >= 5:
            ftest.Late90[i] = 5
    for i in range(len(ftest)):
        if ftest.PropLines[i] >= 6:
            ftest.PropLines[i] = 6
    for i in range(len(ftest)):
        if ftest.Late6089[i] >= 3:
            ftest.Late6089[i] = 3

    ftest.UnsecLines = pd.qcut(ftest.UnsecLines.values, 5).codes
    ftest.age = pd.qcut(ftest.age.values, 5).codes
    ftest.MonthlyIncome = pd.qcut(ftest.MonthlyIncome.values, 5).codes
    ftest.DebtRatio = pd.qcut(ftest.DebtRatio.values, 5).codes
    ftest.OpenCredit = pd.qcut(ftest.OpenCredit.values, 5).codes
    
    ftest = pd.get_dummies(ftest, columns = ["UnsecLines"], prefix="UnsecLines")
    ftest = pd.get_dummies(ftest, columns = ["age"], prefix="age")
    ftest = pd.get_dummies(ftest, columns = ["Late3059"], prefix="Late3059")
    ftest = pd.get_dummies(ftest, columns = ["DebtRatio"], prefix="DebtRatio")
    ftest = pd.get_dummies(ftest, columns = ["MonthlyIncome"], prefix="MonthlyIncome")
    ftest = pd.get_dummies(ftest, columns = ["OpenCredit"], prefix="OpenCredit")
    ftest = pd.get_dummies(ftest, columns = ["Late90"], prefix="Late90")
    ftest = pd.get_dummies(ftest, columns = ["PropLines"], prefix="PropLines")
    ftest = pd.get_dummies(ftest, columns = ["Late6089"], prefix="Late6089")
    ftest = pd.get_dummies(ftest, columns = ["Deps"], prefix="Deps")
    
    Loaded_Model = joblib.load('financial/Saved_Model.pk1')
    
    f = ftest[:1]
    f = f.drop(["Deps_5.0", "Deps_6.0", "Deps_8.0"], axis=1)
    
    score = Loaded_Model.predict_proba(f) #generating score from the loaded model
    score = score[:,1]
    
    return score