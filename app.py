from flask import redirect, url_for, session, g
from database import UserModel
from app_init import *


@app.route('/')
def index():
    return redirect(url_for("user.login"))


@app.before_request
def before_request():
    username = session.get("user")
    if username:
        # 给全局变量g绑定参数
        # setattr(g, "user", user)
        g.user = username


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
