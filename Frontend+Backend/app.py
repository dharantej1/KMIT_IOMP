import joblib
import numpy as np
from flask import Flask,render_template,request,send_file
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import StandardScaler
from joblib import load

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
    data_to_be_predicted = scaler.transform(data_to_be_predicted)
    print("updated data is ",data_to_be_predicted)
    final_result=model.predict(data_to_be_predicted)
    if abs(final_result)>0.5:
        return render_template("waterqualityanalysis.html",pred="The Sample of Water is Suitable for Drinking")
    else:
        return render_template("waterqualityanalysis.html",pred="The Sample of Water is Not Suitable for Drinking")


    # fig,ax=plt.subplots(figsize=(6,6))
    # ax=sns.set_style(style="darkgrid")
    

if __name__ == '__main__':
    app.run(debug=True)