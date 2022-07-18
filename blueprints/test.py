from flask import (Blueprint,
                   render_template)

bp = Blueprint(name="test", import_name=__name__, url_prefix="/test")


@bp.route("/", methods=["GET", "POST"])
def test():
    return render_template("test.html")