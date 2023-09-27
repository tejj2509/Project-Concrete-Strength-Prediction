from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def base():
    return render_template('home.html')

@app.route('/predict', methods = ['post'])
def printall():
    cement = request.form.get("Cement")
    blst_frnc_slg = request.form.get("Blast_Furnace_Slag")
    fly_ash = request.form.get("Fly_Ash")
    water = request.form.get("Water")
    superplasticizer = request.form.get("Superplasticizer")
    coarse_aggr = request.form.get("Coarse_Aggregate")
    fine_aggr = request.form.get("Fine_Aggregate")
    age = request.form.get("Age")

    print(cement, blst_frnc_slg, fly_ash, water, superplasticizer, coarse_aggr, fine_aggr, age)
    model = joblib.load('concrete_strength_81.pkl')
    tf_res = joblib.load('yjtransformer.pkl')

    new_prediction = model.predict(tf_res.transform([[cement, blst_frnc_slg, fly_ash, water, superplasticizer, coarse_aggr, fine_aggr, age]]))
    print("The compressive strength of the concrete structure with the given input features is:",float(new_prediction)) 


    return render_template('predicted.html', data = new_prediction)

app.run(debug=True)