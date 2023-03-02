from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="predict", import_name=__name__, url_prefix="/predict")


@bp.route('/cls', methods=["GET", "POST"])
def predict_cls():
    return render_template("templates_cls/predict.html")

@bp.route('/seg', methods=["GET", "POST"])
def predict_seg():
    return render_template("templates_seg/predict.html")

@bp.route('/det', methods=["GET", "POST"])
def predict_det():
    return render_template("templates_det/predict.html")

@bp.route('/fd', methods=["GET", "POST"])
def predict_fd():
    return render_template("templates_fd/predict.html")