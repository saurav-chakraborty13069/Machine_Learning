from flask import Flask, render_template, flash, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from forms import PredictionForm
import pickle


app = Flask(__name__)
app.config['SECRET_KEY'] = '19416c21be3d611b5d01b37409a55fe9'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediction.db'
db = SQLAlchemy(app)


class Flower(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sl = db.Column(db.Float())
    sw = db.Column(db.Float())
    pl = db.Column(db.Float())
    pw = db.Column(db.Float())

    def __repr__(self):
        return f"Flower('{self.sl}','{self.sw}','{self.pl}','{self.pw}')"
        # return f"User('{self.username}', '{self.email}', '{self.image_file}')"


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    form = PredictionForm()
    if form.validate_on_submit():
        flower = Flower(sl=form.sl.data, sw=form.sw.data, pl=form.pl.data, pw=form.pw.data)
        db.session.add(flower)
        db.session.commit()
        return redirect(url_for('predict'))
    return render_template('data.html', form=form)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # flower = Flower.query.all()[-1]
    # return render_template('predict.html')
    flower = Flower.query.all()[-1]
    test_data = [[flower.sl, flower.sw, flower.pl, flower.pw]]
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(test_data)
    output = prediction[0]
    return render_template('predict.html', value=output)


if __name__ == '__main__':
    app.run(port = 5001, debug=True)
