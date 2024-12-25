from flask import Flask,render_template,request
import pandas as pd
import pickle
app = Flask(__name__)   # Flask constructor 
data = pd.read_csv("Cleaned_car.csv")
model = pickle.load(open("model.pk","rb"))
companylist = sorted(data["company"].unique())
yearlist  = sorted(data["year"].unique())
kmslist = sorted(data["kms_driven"].unique())
fuellist = sorted(data["fuel_type"].unique())

@app.route('/')    
def index(): 
    return render_template("index.html",company = companylist,year = yearlist,kms = kmslist,fuel = fuellist)

@app.route('/pred',methods=["GET"])    

def Predict():
    if request.method =="GET":
        company =request.args.get("com")
        year = int(request.args.get("year"))
        kms = int(request.args.get("kms"))
        fuel = request.args.get("fuel")
        print(company,year,kms,fuel)
        pred = model.predict(pd.DataFrame([[company,year,kms,fuel]],columns=['company','year','kms_driven','fuel_type']))
    return render_template("index.html",pred = int(pred[0]),company = companylist,year = yearlist,kms = kmslist,fuel = fuellist)
    
  
if __name__=='__main__': 
   app.run(debug=True)