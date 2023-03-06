import os

# 必须并行执行
# terminal_train = "streamlit run train_app.py --server.port 8501 --server.fileWatcherType none"
# terminal_upload = "streamlit run upload_image_app.py --server.port 8502 --server.fileWatcherType none"
# terminal_predict = "streamlit run predict_app.py --server.port 8503 --server.fileWatcherType none"

# TERMINALS = [terminal_train, terminal_upload, terminal_predict]
# shell = ""
# for T in TERMINALS:
#     shell += T + '&'

# os.system(shell)

# os.system("streamlit run upload_image_app.py --server.port 8502 --server.fileWatcherType none")

# For testing detection
# os.system("streamlit run detect_predict_app.py --server.port 8504 --server.fileWatcherType none")


os.system(
    "streamlit run upload_image_app.py --server.port 8500 --server.fileWatcherType none &" +
    # 缺陷分类
    "streamlit run cls_train_app.py --server.port 8501 --server.fileWatcherType none &" +
    "streamlit run cls_predict_app.py --server.port 8502 --server.fileWatcherType none &" +
    # 缺陷分割
    "streamlit run seg_predict_app.py --server.port 8602 --server.fileWatcherType none &" +
    # 缺陷检测
    "streamlit run det_predict_app.py --server.port 8702 --server.fileWatcherType none &" +
    # 故障诊断
    "streamlit run fd_predict_app.py --server.port 8801 --server.fileWatcherType none &" +
    "streamlit run fd_preprocessing_app.py --server.port 8802 --server.fileWatcherType none &" +
    # 图像采集
    "streamlit run image_capture_app.py --server.port 8901 --server.fileWatcherType none &"
    )

#