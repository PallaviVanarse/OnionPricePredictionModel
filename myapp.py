import numpy as np
import pickle

from flask import Flask,render_template,request
app=Flask(__name__)

@app.route('/')
def hello_world():
   return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def get_result():
   poly = pickle.load(open('polyo.pkl','rb'))
   model = pickle.load(open('modelo.pkl','rb'))
   query =[[int(request.form['text1']),int(request.form['text2']),int(request.form['text3'])]]
   
   x_query = poly.transform(query)
   modalprice = model.predict(x_query)
   return 'Predicted Modal Price of Onion is Rs. '+str(int(modalprice)) +' per Quintal and Rs. '+str(int((modalprice)/100)) +' per Kg'




if __name__ == '__main__':
   app.run(debug=True)