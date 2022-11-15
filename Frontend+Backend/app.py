import joblib
import numpy as np
from flask import Flask,render_template,request

app = Flask(__name__,template_folder="./")

model=joblib.load("job")


@app.route('/')
def hello_world():
    return render_template("waterqualityanalysis.html",pred="Water Potability Prediction")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    data_to_be_predicted=np.array(int_features,ndmin=2)
    final_result=model.predict(data_to_be_predicted)
    if abs(final_result)>0.5:
        return render_template("waterqualityanalysis.html",pred="The Sample of Water is Suitable for Drinking")
    else:
        return render_template("waterqualityanalysis.html",pred="The Sample of Water is Not Suitable for Drinking")
    

if __name__ == '__main__':
    app.run(debug=True)
