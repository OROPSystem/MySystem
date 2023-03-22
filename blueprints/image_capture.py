from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="image_capture", import_name=__name__, url_prefix="/image_capture")


@bp.route("/ic", methods=["GET", "POST"])
def image_capture():
    return render_template("templates_ic/image_capture.html")