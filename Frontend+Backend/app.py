import joblib
import numpy as np
from flask import Flask,render_template,request,send_file
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from fpdf import FPDF



app = Flask(__name__,template_folder="./")

model=joblib.load("job")


@app.route('/')
def hello_world():
    return render_template("waterqualityanalysis.html",pred="Water Potability Prediction")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    data_to_be_predicted=np.array(int_features,ndmin=2)
    scaler=load('std_scaler.bin')
    print("data is ",data_to_be_predicted)
    data_to_be_predicted_normal = scaler.transform(data_to_be_predicted)
    print("updated data is ",data_to_be_predicted_normal)
    final_result=model.predict(data_to_be_predicted_normal)
    if abs(final_result)>0.5:
        return render_template("waterqualityanalysis.html",pred="The Sample of Water is Suitable for Drinking")
    else:
        return render_template("waterqualityanalysis.html",pred="The Sample of Water is Not Suitable for Drinking")
    
pdf=FPDF()
pdf.add_page()
pdf.set_font("Arial","B",16)
pdf.cell(40, 10, 'Hello World!')
pdf.output('tuto1.pdf', 'F')

if __name__ == '__main__':
    app.run(debug=True)