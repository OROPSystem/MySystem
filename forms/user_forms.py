import wtforms
from wtforms.validators import length, email, EqualTo, InputRequired
from database import UserModel, CaptchaModel, UserSessionModel
from datetime import datetime
from werkzeug.security import check_password_hash

# 定义验证码有效时长，单位：s
CPATCHA_TIME = 60 * 5


# 登录时需要验证的表单
class LoginForm(wtforms.Form):
    username = wtforms.StringField(validators=[length(min=4, max=50, message="用户名长度有误"), InputRequired("用户名不能为空")])
    password = wtforms.PasswordField(validators=[length(min=6, max=200, message="密码长度有误"), InputRequired("密码不能为空")])
    remember = wtforms.BooleanField()

    def validate_username(self, field):
        username = field.data
        user = UserModel.query.filter_by(username=username).first()
        if not user:
            raise wtforms.ValidationError(message="该用户未注册")

    def validate_password(self, field):
        password = field.data
        username = self.username.data
        user = UserModel.query.filter_by(username=username).first()
        if user:
            if user.password != password and (not check_password_hash(user.password, password)):
                raise wtforms.ValidationError(message="密码错误，请重试")


# 注册时需要验证的表单
class RegisterForm(wtforms.Form):
    username = wtforms.StringField(validators=[length(min=4, max=50, message="用户名长度有误"), InputRequired("用户名不能为空")])
    email = wtforms.StringField(validators=[email(message="邮件格式有误"), InputRequired("邮箱不能为空")])
    password = wtforms.PasswordField(validators=[length(min=6, max=50, message="密码长度有误"), InputRequired("密码不能为空")])
    password_confirm = wtforms.PasswordField(
        validators=[EqualTo("password", message="两次密码输入不同"), InputRequired("确认密码不能为空")])
    captcha = wtforms.StringField(validators=[InputRequired("验证码不能为空")])

    # 自定义验证函数，以validate+表单名为函数名会自动调用该验证函数
    def validate_username(self, field):
        username = field.data
        user = UserModel.query.filter_by(username=username).first()
        if user:
            raise wtforms.ValidationError(message="该用户名已被注册")

    # 验证邮箱是否被注册
    def validate_email(self, field):
        email = field.data
        user = UserModel.query.filter_by(email=email).first()
        if user:
            raise wtforms.ValidationError(message="该邮箱已被注册")

    # 先将验证码信息存储到关系数据库中
    # TODO 使用redis或者MangoDB实现验证码缓存
    # 验证验证码是否正确或过期
    def validate_captcha(self, field):
        captcha = field.data
        email = self.email.data
        captcha_model = CaptchaModel.query.filter_by(email=email).first()
        if not captcha_model:
            raise wtforms.ValidationError(message="请先获取验证码")
        else:
            if captcha.lower() != captcha_model.captcha.lower():
                raise wtforms.ValidationError(message="验证码错误")
            elif (datetime.now() - captcha_model.timestamp).total_seconds() > CPATCHA_TIME:
                raise wtforms.ValidationError(message="验证码过期")
