import json

from flask import (Blueprint,
                   render_template,
                   request, redirect,
                   url_for,
                   jsonify,
                   session)
from decorators import login_required

bp = Blueprint(name="data_processing", import_name=__name__, url_prefix="/data_processing")

@bp.route("/", methods=["GET", "POST"])
@login_required
def data_processing_fd():
    # TODO 加入用户信息
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_fd/preprocessing.html",
        zh=zh_json["Dashboard"]
    )