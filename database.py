import datetime

from extensions import db


# 建立存储用户个人信息的数据模型
class UserModel(db.Model):
    __tablename__ = "UserModel"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(200), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)


# 建立存储用户session的数据模型
class UserSessionModel(db.Model):
    __tablename__ = "UserSessionModel"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("UserModel.id"), unique=True)
    session_id = db.Column(db.String(24), unique=True)
    token = db.Column(db.String(50), unique=False)
    timestamp = db.Column(db.TIMESTAMP, unique=False, default=datetime.datetime.now)


# 建立存储用户验证码的数据模型
class CaptchaModel(db.Model):
    __tablename__ = "CaptchaModel"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(200), nullable=False, unique=True)
    captcha = db.Column(db.String(6), nullable=False, unique=False)
    timestamp = db.Column(db.TIMESTAMP, unique=False, default=datetime.datetime.now)


class TabularModel(db.Model):
    __tablename__ = "TabularModel"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), db.ForeignKey("UserModel.username"))
    dataset = db.Column(db.String(200), unique=True, nullable=False)
    train = db.Column(db.String(200), unique=False, nullable=False)
    test = db.Column(db.String(200), unique=False, nullable=True)

    user = db.relationship("UserModel", backref="tabulars")


class ImageModel(db.Model):
    __tablename__ = "ImageModel"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), db.ForeignKey("UserModel.username"))
    dataset = db.Column(db.String(200), unique=True, nullable=False)
    train = db.Column(db.String(200), unique=False, nullable=False)
    test = db.Column(db.String(200), unique=False, nullable=True)

    user = db.relationship("UserModel", backref="images")


class ModelsStorage(db.Model):
    __tablename__ = "ModelsStorage"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model_name = db.Column(db.String(50), unique=True, nullable=False)
    model_whole_path = db.Column(db.String(100), unique=True, nullable=False)
