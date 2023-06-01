from flask import (Blueprint,
                   render_template,
                   redirect,
                   url_for,
                   request)

bp = Blueprint(name="train_orgin", import_name=__name__, url_prefix="/train_orgin")


@bp.route("/run", methods=["GET", "POST"])
def run_cls():
    if request.method == "GET":
        return render_template("templates_cls/train.html")


@bp.route("/tensorboard", methods=["GET", "POST"])
def tensorboard_cls():
    return render_template("templates_cls/tensorboard.html")

@bp.route("/train_fd", methods=["GET", "POST"])
def train_fd():
    return render_template("templates_fd/train.html")
