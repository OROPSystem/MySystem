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


@bp.route("/")
def models_storage():
    return render_template("models_storage.html")


@bp.route("/quote_model", methods=["POST"])
def quote_model():
    quote_name = request.form.to_dict()["quote_name"]
    model = ModelsStorage.query.filter_by(model_name=quote_name).first()
    source_file = pathlib.Path(f"models_storage/{model.model_whole_path}")
    # TODO 将username存储到redis
    target_file = pathlib.Path(f"user_data/test/models")
    shutil.copy(source_file, target_file)
    return {"code": 200}
