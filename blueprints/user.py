import datetime
from flask import (Blueprint,
                   render_template,
                   request, redirect,
                   url_for,
                   jsonify,
                   session,
                   g)
from flask_mail import Message
from forms import LoginForm, RegisterForm
from database import UserModel, UserSessionModel, CaptchaModel
from extensions import mail, db
from global_variable import USER_ROOT, local_session, appConfig
import string
import random
from werkzeug.security import generate_password_hash
from oldutils import sys_ops

bp = Blueprint(name="user", import_name=__name__, url_prefix="/user")


# 用户的登录视图函数
@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        print(session.get("token"))
        # if session.get("token") is not None:
        #     user_session = UserSessionModel.query.filter_by(token=session["token"]).first()
        #     # TODO 正常来说应该不会没有session database，仅在测试时使用
        #     if user_session:
        #         user = UserModel.query.filter_by(id=user_session.user_id).first()
        #         content = {
        #             'username': user.username,
        #             'password': user.password,
        #             'remember': 1
        #         }
        #         print("记住token了")
        #         return render_template("login_register.html", **content)
        #     else:
        #         return render_template("login_register.html")
        # else:
        #     print("没记住")
        return render_template("login_register.html")
    else:
        form = LoginForm(request.form)
        # if form.validate():
        username = form.username.data
        # user = UserModel.query.filter_by(username=username).first()
        session["user"] = username
        #     if form.remember.data and (session.get("token") is None):
        #         print("重新生成token")
        #         session.permanent = True
        #         letters = string.ascii_letters + string.digits
        #         token = "".join(random.sample(letters, 32))
        #         session["token"] = token
        #         user_session = UserSessionModel.query.filter_by(user_id=user.id).first()
        #         if user_session:
        #             user_session.token = token
        #             user_session.timestamp = datetime.datetime.now()
        #             db.session.commit()
        #         else:
        #             user_session = UserSessionModel(user_id=user.id, token=token)
        #             db.session.add(user_session)
        #             db.session.commit()
        #     elif (session.get("token") is not None) and (not form.remember.data):
        #         session.pop("token")
        return redirect(url_for('dashboard.dashboard'))
        # else:
        #     return render_template("login_register.html", form=form)


@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register_login.html")
    else:
        form = RegisterForm(request.form)
        if form.validate():
            username = form.username.data
            email = form.email.data
            password = form.password.data
            password = generate_password_hash(password)
            user = UserModel(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()
            session["user"] = username
            sys_ops.create_user_path(USER_ROOT, username, local_session, session, appConfig)
            return redirect(url_for("user.login"))
        else:
            return render_template("register_login.html", form=form)


@bp.route('/get_captcha', methods=["GET", "POST"])
def get_captcha():
    letters = string.ascii_letters + string.digits
    captcha = "".join(random.sample(letters, 6))

    email = request.json["email"]

    captcha_model = CaptchaModel.query.filter_by(email=email).first()
    # TODO 只有注册和找回密码使用验证码，不需要用数据库
    if captcha_model:
        captcha_model.captcha = captcha
        captcha_model.timestamp = datetime.datetime.now()
        db.session.commit()
    else:
        captcha_model = CaptchaModel(captcha=captcha, email=email)
        db.session.add(captcha_model)
        db.session.commit()
    message = Message(
        subject="逼乎",
        recipients=[email],
        body=f"你的验证码为：{captcha}",
    )
    # mail.send(message)
    print(f"captcha: {captcha}")
    return jsonify({"message": "成功发送"})
