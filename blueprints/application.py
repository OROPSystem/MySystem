from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="application", import_name=__name__, url_prefix="/application")


@bp.route('/', methods=["GET", "POST"])
def application():
    return render_template("application.html")