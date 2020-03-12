from flask_wtf import FlaskForm


from wtforms import DecimalField, SubmitField
from wtforms.validators import DataRequired, NumberRange
# from flask_blog.models import User


class PredictionForm(FlaskForm):
    sl = DecimalField('Sepal Length', validators=[DataRequired(), NumberRange(min=0, max=10)])
    sw = DecimalField('Sepal Width', validators=[DataRequired(), NumberRange(min=0, max=10)])
    pl = DecimalField('Petal Length', validators=[DataRequired(), NumberRange(min=0, max=10)])
    pw = DecimalField('Petal Width', validators=[DataRequired(), NumberRange(min=0, max=10)])

    submit = SubmitField('Predict')
