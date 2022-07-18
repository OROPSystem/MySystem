import streamlit as st
import zipfile, tarfile, rarfile
import pathlib
import numpy as np
import h5py

compressed_method = {"zip": zipfile.ZipFile,
                     "tar": tarfile.TarFile,
                     "rar": rarfile.RarFile}


class FolderPerClass:
    def __init__(self, dataset_file_name):
        self.dataset_file_name = dataset_file_name
        self.ext = self.dataset_file_name.split(".")[-1]
        self.pre = self.dataset_file_name.split(".")[0]

    def to_h5(self, compressed_file):
        decompress = compressed_method[self.ext]
        decompress_file = decompress(compressed_file, "r")
        dataset_dir = pathlib.Path(f"./user_data/test/datasets/image/{self.pre}")
        dataset_dir.mkdir(exist_ok=True)
        decompress_file.extractall(dataset_dir)
        decompress_file.close()
        for inner_dir in dataset_dir.iterdir():
            for data_class in inner_dir.iterdir():
                st.write(data_class)
        pass


class NumpyFile:
    def __init__(self, dataset_file_name):
        self.dataset_file_name = dataset_file_name
        self.ext = self.dataset_file_name.split(".")[-1]

    def to_h5(self, compressed_file):
        pass


class FolderWithLabelFile:
    def __init__(self, dataset_file_name):
        self.dataset_file_name = dataset_file_name
        self.ext = self.dataset_file_name.split(".")[-1]

    def to_h5(self, compressed_file):
        pass


dataset_name = st.sidebar.text_input(label="Dataset Name", placeholder="不要包含空格")
dataset_format = st.sidebar.selectbox(label="Dataset Format",
                                      options=["Folder per class", "All same folder with label file", "Numpy file"])
show_dataset_format_sketch = st.sidebar.checkbox(label="show dataset format sketch", value=False)

dataset_file = st.sidebar.file_uploader(label="Dataset", help="还没想好", accept_multiple_files=False, type=["zip", "rar", "tar"])

if dataset_file is not None:
    dataset_bytes = dataset_file.getvalue()
    # TODO 增加user变量
    temporary_file = pathlib.Path(f"./user_data/test/datasets/image/{dataset_file.name}")
    temporary_file.touch(exist_ok=True)
    temporary_file.write_bytes(dataset_bytes)
    dataset = FolderPerClass(dataset_file.name)
    dataset.to_h5(temporary_file)
    st.sidebar.write("完成")

if show_dataset_format_sketch:
    st.subheader("Dataset Format Sketch", anchor=None)
    if dataset_format == "Folder per class":
        st.text('my-images/\n' +
                '├── train_orgin/\n' +
                '│   ├── class1/\n' +
                '│   │   └── *.jpg/.png\n' +
                '│   └── class2/\n' +
                '│       └── *.jpg/.png\n' +
                '└── test/\n' +
                '    ├── class1/\n' +
                '    │   └── *.jpg/.png\n' +
                '    └── class2/\n' +
                '        └── *.jpg/.png')
    elif dataset_format == "All same folder with label file":
        st.text('my-images/\n' +
                '├── images/\n'
                '│   ├── 1.jpg/.png\n' +
                '│   ├── 2.jpg/.png\n' +
                '│   └── 3.jpg/.png\n' +
                '├── train_orgin.txt/.csv\n' +
                '└── test.txt/.csv')
    elif dataset_format == "Numpy file":
        st.text('Save your data with:\n\n' +
                'np.savez(filename, x=images, y=labels)\n\n' +
                'or\n\n' +
                'np.savez(filename, x_train=images_train, y_train=labels_train, x_test=images_test,   y_test=labels_test)')
    st.markdown("---")

st.subheader("")
