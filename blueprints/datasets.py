from flask import (Blueprint,
                   render_template,
                   redirect,
                   url_for,
                   request,
                   jsonify,
                   session,
                   g)
from forms import NewTabularFileForm, NewImageFileForm
from database import TabularModel, UserModel, ImageModel
from extensions import db
from global_variable import USER_ROOT
from decorators import login_required
from oldutils import config_ops, upload_util
from utils import user_data_ops

bp = Blueprint(name="datasets", import_name=__name__, url_prefix="/datasets")


@bp.route("/tabular", methods=["GET", "POST"])
@login_required
def tabular():
    if request.method == "GET":
        username = session['user']
        datasets = TabularModel.query.filter_by(username=username)
        examples = upload_util.get_examples()
        return render_template("upload_tabular.html",
                               datasets=datasets,
                               # dataset_types=dataset_types,
                               examples=examples)
    return render_template("upload_tabular.html")


@bp.route("/image", methods=["GET", "POST"])
@login_required
def image():
    if request.method == "GET":
        username = session['user']
        datasets = ImageModel.query.filter_by(username=username)
        return render_template("upload_image.html",
                               datasets=datasets)


# 保存upload表单数据
@bp.route("/save_tabular", methods=["POST"])
def save_tabular():
    if request.method == "POST":
        files = request.files
        # upload Data
        # 将dataset地址存储到数据库
        form = NewTabularFileForm(files)
        if form.validate():
            dataset_name = form.train_file.data.filename.split('.')[0]
            # 将dataset以文件的形式存储到磁盘
            user_data_ops.create_tabular(USER_ROOT, username=g.user, datasetname=dataset_name,
                                         train=files.get("train_file"), test=files.get("test_file"))
            datasets = TabularModel(username=g.user, dataset=dataset_name, train=dataset_name)
            db.session.add(datasets)
            db.session.commit()
            return jsonify({"code": 200,
                            "message": "上传成功！"})
        else:
            return jsonify({"code": 400,
                            "train_file": form.train_file.errors,
                            "test_file": form.test_file.errors})


# 保存generate表单数据
@bp.route("/save_generate", methods=["GET", "POST"])
def save_generate():
    form = request.form
    print(form)
    return jsonify({"code": 200,
                    "message": "上传成功！"})


@bp.route("/save_image", methods=["POST"])
def save_image():
    if request.method == "POST":
        files = request.files
        # upload Data
        # 将dataset地址存储到数据库
        form = NewImageFileForm(files)
        if form.validate():
            dataset_name = form.image_file.data.filename.split('.')[0]
            # 将dataset以文件的形式存储到磁盘
            user_data_ops.create_image(USER_ROOT, username=g.user, datasetname=dataset_name,
                                       image=files.get("image_file"))
            dataset = ImageModel(username=g.user, dataset=dataset_name, train=dataset_name)
            db.session.add(dataset)
            db.session.commit()
            return jsonify({"code": 200,
                            "message": "上传成功！"})
        else:

            return jsonify({"code": 400,
                            "image_file": form.image_file.errors})
