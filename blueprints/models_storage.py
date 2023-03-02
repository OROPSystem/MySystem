import json
import pathlib

from flask import (Blueprint,
                   render_template,
                   redirect,
                   url_for,
                   request,
                   jsonify,
                   g)
from decorators import login_required
from utils import create_pytorch_model
from database import ModelsStorage
import pathlib
import shutil

bp = Blueprint(name="models_storage", import_name=__name__, url_prefix="/models_storage")


@bp.route("/cls")
def models_storage_cls():
    return render_template("templates_cls/models_storage.html")

@bp.route("/seg")
def models_storage_seg():
    return render_template("templates_seg/models_storage.html")

@bp.route("/det")
def models_storage_det():
    return render_template("templates_det/models_storage.html")

@bp.route("/fd")
def models_storage_fd():
    return render_template("templates_fd/models_storage.html")

@bp.route("/quote_model", methods=["POST"])
def quote_model():
    quote_name = request.form.to_dict()["quote_name"]
    model = ModelsStorage.query.filter_by(model_name=quote_name).first()
    source_file = pathlib.Path(f"models_storage/{model.model_whole_path}")
    # TODO 将username存储到redis
    target_file = pathlib.Path(f"user_data/test/models")
    shutil.copy(source_file, target_file)
    return {"code": 200}
