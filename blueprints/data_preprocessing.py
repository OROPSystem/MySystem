from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="data_preprocessing", import_name=__name__, url_prefix="/data_preprocessing")


@bp.route("/fd", methods=["GET", "POST"])
def data_preprocessing_fd():
    return render_template("templates_fd/preprocessing.html")

