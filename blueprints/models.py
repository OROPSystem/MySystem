import json
import pathlib

from flask import (Blueprint,
                   render_template,
                   redirect,
                   url_for,
                   request,
                   jsonify)
from decorators import login_required
from utils import create_pytorch_model

bp = Blueprint(name="models", import_name=__name__, url_prefix="/models")

@bp.route("/cls", methods=["GET", "POST"])
@login_required
def models_cls():
    return render_template("templates_cls/model.html")


@bp.route("/save_model", methods=["POST"])
def save_model():
    form_data = request.form.to_dict()
    model_name = form_data.get("model_name")
    model = json.loads(form_data.get("model"))
    # print(model_name)

    # 根据model name写入相应的json文件
    model_path = pathlib.Path(f"user_data/test/models/{model_name}")
    model_path.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(f"user_data/test/models/{model_name}/{model_name}.json")
    model_path.touch(exist_ok=True)
    with model_path.open("w", encoding="utf-8") as m:
        json.dump(model, m)
    # TODO 将上传的model存储到用户数据库中

    # TODO 用pytorch验证并构建模型

    return jsonify({"message": "success"})
