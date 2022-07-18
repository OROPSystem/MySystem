import json

from flask import (Blueprint,
                   render_template,
                   request, redirect,
                   url_for,
                   jsonify,
                   session)
from decorators import login_required

bp = Blueprint(name="dashboard", import_name=__name__, url_prefix="/dashboard")


@bp.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    # TODO 加入用户信息
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "dashboard.html",
        zh=zh_json["Dashboard"]
        # title="Dashboard",
        # user=username,
        # user_configs=config_ops.get_datasets(USER_ROOT, username),
        # token=get_token_user(username),
    )
