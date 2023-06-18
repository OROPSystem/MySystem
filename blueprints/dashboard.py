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
    )

@bp.route("/application", methods=["GET", "POST"])
@login_required
def dashboard_application():
    # TODO 加入用户信息
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "application.html",
        zh=zh_json["Dashboard"]
    )

@bp.route("/cls", methods=["GET", "POST"])
@login_required
def dashboard_cls():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_cls/dashboard.html",
        zh=zh_json["Dashboard"]
    )


@bp.route("/seg", methods=["GET", "POST"])
@login_required
def dashboard_seg():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_seg/dashboard.html",
        zh=zh_json["Dashboard"]
    )

@bp.route("/det", methods=["GET", "POST"])
@login_required
def dashboard_det():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_det/dashboard.html",
        zh=zh_json["Dashboard"]
    )

@bp.route("/fd", methods=["GET", "POST"])
@login_required
def dashboard_fd():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_fd/dashboard.html",
        zh=zh_json["Dashboard"]
    )

@bp.route("/ic", methods=["GET", "POST"])
@login_required
def dashboard_ic():
    # username = session["user"]
    with open("language/text-zh.json", "r", encoding="utf-8") as f:
        zh_json = json.load(f)
    return render_template(
        "templates_ic/dashboard.html",
        zh=zh_json["Dashboard"]
    )