from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="predict", import_name=__name__, url_prefix="/predict")


@bp.route('/', methods=["GET", "POST"])
def predict():
    return render_template("predict.html")