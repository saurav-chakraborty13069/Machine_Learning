import numpy as np
from flask import Flask, request, jsonify, render_template,flash, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from forms import PredictionForm
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model


app = Flask(__name__)
#model = load_model('model.h5')

app = Flask(__name__)
app.config['SECRET_KEY'] = '19416c21be3d611b5d01b37409a55fe9'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
db = SQLAlchemy(app)

#class Flower(db.Model):
class Promotion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dep = db.Column(db.String())
    reg = db.Column(db.Integer())
    edu = db.Column(db.String())
    gen = db.Column(db.String())
    rec = db.Column(db.String())
    trn = db.Column(db.Integer())
    age = db.Column(db.Integer())
    rat = db.Column(db.Integer())
    srv = db.Column(db.Integer())
    kpi = db.Column(db.Integer())
    awd = db.Column(db.Integer())
    scr = db.Column(db.Integer())


    def __repr__(self):
        return f"Promotion('{self.dep}','{self.reg}','{self.edu}','{self.gen}','{self.rec}','{self.trn}','{self.age}','{self.rat}','{self.srv}','{self.kpi}','{self.awd}','{self.scr}')"
        # return f"User('{self.username}', '{self.email}', '{self.image_file}')"

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')
    #return "hello World"

@app.route('/data', methods=['GET', 'POST'])
def data():
    form = PredictionForm()
    if form.validate_on_submit():
        #flower = Flower(sl=form.sl.data, sw=form.sw.data, pl=form.pl.data, pw=form.pw.data)
        promote = Promotion(dep=form.dep.data, reg=form.reg.data, edu=form.edu.data, gen=form.gen.data,
                            rec=form.rec.data, trn=form.trn.data, age=form.age.data, rat=form.rat.data,
                            srv=form.srv.data, kpi=form.kpi.data, awd=form.awd.data, scr=form.scr.data)
        db.session.add(promote)
        db.session.commit()
        return redirect(url_for('predict'))
    return render_template('data.html', form=form)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # flower = Flower.query.all()[-1]
    # return render_template('predict.html')
    #flower = Flower.query.all()[-1]
    promote = Promotion.query.all()[-1]
    #test_data = [[promote.dep, promote.reg, promote.edu, promote.gen,
    #                promote.rec, promote.trn, promote.age, promote.rat
    #                promote.srv, promote.kpi, promote.awd, promote.scr]]

    test_data = {'Department':promote.dep,
                    'Region':promote.reg,
                    'Education':promote.edu,
                    'Gender':promote.gen,
                    'Recruitment Source':promote.rec,
                    'Trainings':promote.trn,
                    'Age':promote.age,
                    'Previous Rating': promote.rat,
                    'Service years': promote.srv,
                    'KPI': promote.kpi,
                    'Awards': promote.awd,
                    'Score': promote.scr}

    df = pd.DataFrame(test_data)
    data = df.iloc[:,:].values

    labelencoder1 = LabelEncoder()
    data[:,0] = labelencoder1.fit_transform(data[:,0])
    labelencoder2 = LabelEncoder()
    data[:,2] = labelencoder2.fit_transform(data[:,2])
    labelencoder3 = LabelEncoder()
    data[:,3] = labelencoder3.fit_transform(data[:,3])
    labelencoder4 = LabelEncoder()
    data[:,4] = labelencoder3.fit_transform(data[:,4])
    onehotencoder1 = OneHotEncoder(categorical_features = [0, 2, 4])
    data = onehotencoder1.fit_transform(data).toarray()
    
    
    #model = pickle.load(open('model.pkl', 'rb'))
    model = load_model('model.h5')
    prediction = model.predict(data)
    out = prediction[0]
    if out == 0:
        output = "No Promotion, Regrets!!!"
    else:
        output = "Promotion"

    return render_template('predict.html', value=output)


if __name__ == '__main__':
    app.run(port = 4003, debug=True)