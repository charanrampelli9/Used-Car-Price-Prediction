### Intregrate HTML with flask 
### HTTP VERB get ASND pOST
# .venv\Scripts\Activate
'''
{%...%} conditions for,if
{{}} expressions to print output
{#...#} this is for comments
'''
import pandas as pd
from flask import Flask,url_for,redirect,render_template,request
from ced19i041_hcl_project import linearRegression
from Transformer import Transformer
app = Flask(__name__)
@app.route("/") #decorator
def home():
    return render_template('form.html')

# @app.route("/results/<int:score>") #decorator
@app.route("/submit/results/<res>") #decorator
def results(res):
    return render_template('results.html',print_statement=res)

@app.route('/submit',methods=['POST','GET'])
def submit():
    res_sending="out"
    if request.method == 'POST':
        Brand=(request.form['Brand'])
        Loaction=(request.form['Location'])
        Year=(request.form['Year'])
        Kilometers_Driven=(request.form['Kilometers_Driven'])
        Fuel_Type=(request.form['Fuel_Type'])
        Transmission=(request.form['Transmission'])
        Owner_Type=(request.form['Owner_Type'])
        Mileage=(request.form['Mileage'])
        Engine=(request.form['Engine'])
        Power=(request.form['Power'])
        Seats=(request.form['Seats'])
    df={"Year":Year,'Kilometers_Driven':Kilometers_Driven,"Fuel_Type":Fuel_Type,
        'Transmission':Transmission,
        "Owner_Type":Owner_Type,'Mileage':Mileage,
        'Engine':Engine,'Power':Power,'Seats':Seats,"Brand":Brand}
    X_test=pd.DataFrame(df,index=[0])
    res_sending=Transformer(X_test)         
    return redirect(url_for('results',res=res_sending))
    
if __name__ =='__main__':
    app.run(debug=True)