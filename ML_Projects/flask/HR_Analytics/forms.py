from flask_wtf import FlaskForm
from wtforms import DecimalField, SubmitField, StringField, IntegerField
from wtforms.validators import DataRequired, NumberRange, Length

class PredictionForm(FlaskForm):
    #sl = DecimalField('Sepal Length', validators=[DataRequired(), NumberRange(min=0, max=10)])
    #sw = DecimalField('Sepal Width', validators=[DataRequired(), NumberRange(min=0, max=10)])
    #pl = DecimalField('Petal Length', validators=[DataRequired(), NumberRange(min=0, max=10)])
    #pw = DecimalField('Petal Width', validators=[DataRequired(), NumberRange(min=0, max=10)])


    dep = StringField('Department', validators=[DataRequired(), Length(min = 2, max = 15)])
    reg = IntegerField('Region', validators=[DataRequired(), NumberRange(min = 0, max = 9)])
    edu = StringField('Education', validators=[DataRequired(), Length(min = 2, max = 15)])
    gen = StringField('Gender', validators=[DataRequired(), Length(min = 1, max = 15)])
    rec = StringField('Recruitment Source', validators=[DataRequired(), Length(min = 1, max = 15)])
    trn = IntegerField('Trainings', validators=[DataRequired(), NumberRange(min = 1, max = 9)])
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min = 0, max =99)])
    rat = IntegerField('Previous Year Rating', validators=[DataRequired(), NumberRange(min = 1, max = 5)])
    srv = IntegerField('Length of Service', validators=[DataRequired(), NumberRange(min = 1, max = 35)])
    kpi = IntegerField('KPI Met', validators=[DataRequired(), NumberRange(min = 0, max = 1)])
    awd = IntegerField('Awards', validators=[DataRequired(), NumberRange(min = 0, max = 1)])
    scr = IntegerField('Average Training Score', validators=[DataRequired(), NumberRange(min = 0, max = 100)])

    submit = SubmitField('Predict')