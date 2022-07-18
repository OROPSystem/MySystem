from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="deployment", import_name=__name__, url_prefix="/deployment")


@bp.route("/", methods=["GET", "POST"])
def deployment():
    return render_template("deploy.html")