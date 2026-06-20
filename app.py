from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():

    if request.method=='GET':
        return render_template('home.html')

    else:

        data=CustomData(

            gender=request.form.get('gender'),
            SeniorCitizen=int(request.form.get('SeniorCitizen')),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            tenure=int(request.form.get('tenure')),
            PhoneService=request.form.get('PhoneService'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamingTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            MonthlyCharges=float(request.form.get('MonthlyCharges')),
            TotalCharges=float(request.form.get('TotalCharges'))

        )


        pred_df=data.get_data_as_data_frame()

        print(pred_df)

        print("Before Prediction")


        predict_pipeline=PredictPipeline()

        print("Prediction Started")


        results=predict_pipeline.predict(pred_df)

        print("Prediction Completed")


        if results[0]==1:
            result="Customer will Churn"
        else:
            result="Customer will not Churn"


        return render_template(
            'home.html',
            results=result
        )


if __name__=="__main__":

    app.run(host="0.0.0.0")