import datetime

from blueprints import (user_bp,
                        dashboard_bp,
                        datasets_bp,
                        models_bp,
                        train_bp,
                        predict_bp,
                        test_bp,
                        deployment_bp,
                        models_storage_bp,
                        application_bp,
                        data_preprocessing_bp,
                        image_capture_bp)
from flask_bootstrap import Bootstrap
from flask_migrate import Migrate
from extensions import db, mail, app
from global_variable import appConfig
from app_config import config

app.config.from_object(config)
# 加载蓝图
app.register_blueprint(user_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(datasets_bp)
app.register_blueprint(models_bp)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(test_bp)
app.register_blueprint(deployment_bp)
app.register_blueprint(models_storage_bp)
app.register_blueprint(application_bp)
app.register_blueprint(data_preprocessing_bp)
app.register_blueprint(image_capture_bp)
Bootstrap(app)

# 设置密钥，用于hash加密
app.secret_key = appConfig.secret_key()
# 数据库配置
app.config["SQLALCHEMY_DATABASE_URI"] = appConfig.database_uri()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = appConfig.track_modifications()
# 不对JSON进行排序，原样返回
app.config["JSON_SORT_KEYS"] = appConfig.json_sort_keys()
# 设置session的到期时间
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(days=7)

# 不对表单进行SCRF保护
# TODO 应该要进行保护吧，到时候加一下
app.config["WTF_CSRF_ENABLED"] = False

db.init_app(app)
migrate = Migrate(app, db)
mail.init_app(app)
