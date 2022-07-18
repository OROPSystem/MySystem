import configparser
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "app_config.ini")

SQLALCHEMY = "SQLALCHEMY"
FLASK = "FLASK"
APP = "APP"
PARAMS = "DEFAULT_PARAMS"
PATHS = "PATHS"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class ConfigApp(object):
    def __init__(self):
        self.config = configparser.ConfigParser()
        # 读取app_config.ini中的配置
        self.config.read(CONFIG_PATH)

    def get(self, section, param):
        """
        获取section中param的配置内容
        """
        return self.config.get(section, param)

    def user_root(self):
        # TODO 完成注释
        if "USER_ROOT" in os.environ:
            return os.environ["USER_ROOT"]

        user_root = self.get(PATHS, "USER_ROOT")

        if user_root != "None" and not os.path.isdir(os.path.join(user_root)):
            os.makedirs(os.path.join(user_root))

        if user_root != "None":
            return user_root

        return None

    def database_uri(self):
        # 判断环境变量或系统变量中是否设置了相关属性
        if (
                "HOSTNAME" in os.environ
                and "DATABASE" in os.environ
                and "USERNAME" in os.environ
                and "PASSWORD" in os.environ
                and "PORT" in os.environ
        ):
            USERNAME = os.environ["USERNAME"]
            PASSWORD = os.environ["PASSWORD"]
            DATABASE = os.environ["DATABASE"]
            HOSTNAME = os.environ["HOSTNAME"]
            PORT = os.environ["PORT"]
            return f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}"

        # 检查配置文件中是否使用了PostgreSQL
        if self.get(SQLALCHEMY, "POSTGRES_HOSTNAME") not in [None, "None", "none"]:
            print("using postgres db")
            USERNAME = self.get(SQLALCHEMY, "POSTGRES_USERNAME")
            PASSWORD = self.get(SQLALCHEMY, "POSTGRES_PASSWORD")
            DATABASE = self.get(SQLALCHEMY, "POSTGRES_DATABASE")
            HOSTNAME = self.get(SQLALCHEMY, "POSTGRES_HOSTNAME")
            PORT = self.get(SQLALCHEMY, "POSTGRES_PORT")
            return f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}"
        # 检查配置文件中是否使用MySQL
        if self.get(SQLALCHEMY, "MYSQL_HOSTNAME") not in [None, "None", "none"]:
            print("using mysql db")
            USERNAME = self.get(SQLALCHEMY, "MYSQL_USERNAME")
            PASSWORD = self.get(SQLALCHEMY, "MYSQL_PASSWORD")
            DATABASE = self.get(SQLALCHEMY, "MYSQL_DATABASE")
            HOSTNAME = self.get(SQLALCHEMY, "MYSQL_HOSTNAME")
            PORT = self.get(SQLALCHEMY, "MYSQL_PORT")
            return f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8"

        # 若果没有配置数据库
        return None

    def track_modifications(self):
        return str2bool(self.get(SQLALCHEMY, "TRACK_MODIFICATIONS"))

    def json_sort_keys(self):
        return str2bool(self.get(FLASK, "JSON_SORT_KEYS"))

    def debug(self):
        if "DEBUG" in os.environ:
            return str2bool(os.environ["DEBUG"])
        return str2bool(self.get(FLASK, "DEBUG"))

    def threaded(self):
        """
        配置多线程
        """
        return str2bool(self.get(FLASK, "THREADED"))

    def host(self):
        return self.get(FLASK, "HOST")

    def port(self):
        return self.get(FLASK, "PORT")

    # def server_name(self):
    #     return self.config.get(FLASK, 'SERVER_NAME')

    # def USER_ROOT(self):
    #     return self.config.get(FLASK, 'APPLICATION_ROOT')

    def sample_data_size(self):
        return int(self.get(APP, "SAMPLE_DATA_SIZE"))

    def max_features(self):
        return int(self.get(APP, "MAX_FEATURES"))

    def max_categorical_size(self):
        return int(self.get(APP, "MAX_CATEGORICAL_SIZE"))

    def max_range_size(self):
        return int(self.get(APP, "MAX_RANGE_SIZE"))

    def min_range_size(self):
        return int(self.get(APP, "MIN_RANGE_SIZE"))

    def num_epochs(self):
        return int(self.get(PARAMS, "num_epochs"))

    def batch_size(self):
        return int(self.get(PARAMS, "batch_size"))

    def optimizer(self):
        return self.get(PARAMS, "optimizer")

    def learning_rate(self):
        return float(self.get(PARAMS, "learning_rate"))

    def throttle(self):
        return int(self.get(PARAMS, "throttle"))

    def save_summary_steps(self):
        return int(self.get(PARAMS, "save_summary_steps"))

    def save_checkpoints_steps(self):
        return int(self.get(PARAMS, "save_checkpoints_steps"))

    def keep_checkpoint_max(self):
        return int(self.get(PARAMS, "keep_checkpoint_max"))

    def secret_key(self):
        """
        设置密钥
        """
        if "SECRET_KEY" in os.environ:
            return os.environ["SECRET_KEY"]
        return "HUSTOROP"


if __name__ == "__main__":
    c = ConfigApp()
    print(c.config.sections())
