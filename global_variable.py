from app_config import config_wrapper
from core.session import Session
import os
from extensions import app

# 创建包含初始化参数和系统配置的对象
appConfig = config_wrapper.ConfigApp()

# 如果配置中有设置USER_ROOT,则使用配置的，否则使用当前目录下的user_data文件夹
# TODO 注释其用途
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
USER_ROOT = (
    appConfig.user_root()
    if appConfig.user_root() is not None
    else os.path.join(APP_ROOT, "user_data")
)

local_session = Session(app, appConfig)
