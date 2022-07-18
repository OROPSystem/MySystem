from flask import (Blueprint,
                   render_template,
                   redirect,
                   url_for,
                   request)

bp = Blueprint(name="train_orgin", import_name=__name__, url_prefix="/train_orgin")


@bp.route("/run", methods=["GET", "POST"])
def run():
    if request.method == "GET":
        return render_template("train.html")


@bp.route("/tensorboard", methods=["GET", "POST"])
def tensorboard():
    return render_template("tensorboard.html")
