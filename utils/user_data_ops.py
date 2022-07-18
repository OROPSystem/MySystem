import pathlib
import gzip
import tarfile
import zipfile
import rarfile

import pandas as pd


# 创建上传的tabular文件
def create_tabular(USER_ROOT, username, datasetname, train, test=None):
    path = pathlib.Path(USER_ROOT)
    path = path / username / "datasets" / "tabular" / datasetname
    path.mkdir(parents=True, exist_ok=True)
    train_path = path / "train_orgin"
    train_path.mkdir(exist_ok=True)
    train_path = train_path / train.filename
    train_path.touch(exist_ok=True)
    train_path.write_bytes(train.read())
    if test:
        # 先创建文件夹，再写入文件
        test_path = path / "test"
        test_path.mkdir(exist_ok=True)
        test_path = test_path / test.filename
        test_path.touch(exist_ok=True)
        test_path.write_bytes(test.read())


def create_image(USER_ROOT, username, datasetname, image):
    path = pathlib.Path(USER_ROOT)
    path = path / username / "datasets" / "image" / datasetname
    path.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path / "train_orgin").mkdir(exist_ok=True)
    pathlib.Path(path / "test").mkdir(exist_ok=True)
    pathlib.Path(path / image.filename).touch(exist_ok=True)
    pathlib.Path(path / image.filename).write_bytes(image.read())
    # 分别解压不同格式的压缩文件
    ext = image.filename.split(".")[-1]
    if ext == "rar":
        rar = rarfile.RarFile(path / image.filename)
        rar.extractall(path / "train_orgin")
        rar.close()
    elif ext == "zip":
        zip_file = zipfile.ZipFile(path / image.filename)
        zip_file.extractall(path / "train_orgin")
        zip_file.close()
    elif ext == "tar":
        tar = tarfile.TarFile(path / image.filename)
        tar.extractall(path / "train_orgin")
        tar.close()
    elif ext == "gz":
        pass
    pass
