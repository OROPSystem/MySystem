from flask import g
import wtforms
from flask_wtf.file import FileAllowed
from wtforms.validators import InputRequired
from database import TabularModel, ImageModel


class NewTabularFileForm(wtforms.Form):
    train_file = wtforms.FileField(validators=[FileAllowed(['csv'], message="请传入CSV格式文件"),
                                               InputRequired(message="训练文件不能为空")])
    test_file = wtforms.FileField(validators=[FileAllowed(['csv'], message="请传入CSV格式文件")])

    def validate_train_file(self, field):
        dataset_name = field.data.filename.split('.')[0]
        dataset = TabularModel.query.filter_by(username=g.user, dataset=dataset_name).first()
        if dataset:
            raise wtforms.ValidationError(message="该数据集已经存在")


class NewImageFileForm(wtforms.Form):
    image_file = wtforms.FileField(validators=[FileAllowed(['zip'], message="请传入正确格式的文件"),
                                               InputRequired(message="文件不能为空")])

    def validate_image_file(self, field):
        dataset_name = field.data.filename.split('.')[0]
        dataset = ImageModel.query.filter_by(username=g.user, dataset=dataset_name).first()
        if dataset:
            raise wtforms.ValidationError(message="该数据集已存在")

    pass
