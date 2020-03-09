from flask import Flask, request, jsonify, render_template, redirect, url_for
from financial.financialScoring import financialScore as fS
from behavioural.behaviouralScore import Behavoural_analysis as bs

app = Flask(__name__)
app.debug = True

financial_score = 0
social_score = 0
user_token = 0
behavioural_score = 0

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        global user_token
        global behavioural_score
        token_from_user = request.form.get("userToken")
        user_token = token_from_user
        # Optinal token verify
        # Put calling function for facebook
        behavioural_score = bs(user_token)
        return redirect(url_for('financialScore'))
    else:
        return render_template('home.html')

@app.route('/financialScore',methods=['POST', 'GET'])
def financialScore():

    if request.method == 'POST':
        dic = {}
        dic["UnsecLines"] = int(request.form.get("loan_balance")) / int(request.form.get("credit_limit_offered"))
        dic["age"] = int(request.form.get("age"))
        dic["Late3059"] = int(request.form.get("default3059"))
        dic["DebtRatio"] = (int(request.form.get("monthly_loan_payments")) + int(request.form.get("monthly_living_cost")))/int(request.form.get("monthly_gross_income"))
        dic["MonthlyIncome"] = int(request.form.get("monthly_gross_income"))
        dic["OpenCredit"] = int(request.form.get("number_of_loans_taken"))
        dic["Late90"] = int(request.form.get("default90"))
        dic["PropLines"] = int(request.form.get("number_of_home_loans_taken"))
        dic["Late6089"] = int(request.form.get("default6089"))
        dic["Deps"] = int(request.form.get("numberOfDependents"))
        output = fS(dic)
        global financial_score
        financial_score = output

        # return render_template('index.html', prediction_text='Default Probability is {}'.format(output))
        return redirect(url_for('result'))
    else:
        return render_template('index.html')

def category(num):
    if num > 0.75:
        return 'Extremely Good'
    elif num > 0.5 and num <= 0.75:
        return 'Good'
    elif num > 0 and num <= 0.5:
        return 'Okay'
    else:
        return 'Concerning'

@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        global financial_score
        global behavioural_score
        score = ((1 - financial_score) + behavioural_score) / 2
        categoryStr = category(score)
        return render_template('result.html', analysis_score='You\'re given a score of {} out of 1'.format(score), score_category='You are put into {} category.'.format(categoryStr))
    else:
        return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)