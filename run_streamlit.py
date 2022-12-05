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


# os.system("streamlit run upload_image_app.py --server.port 8500 --server.fileWatcherType none & " +
#           "streamlit run cls_train_app.py --server.port 8501 --server.fileWatcherType none & " +
#           "streamlit run cls_predict_app.py --server.port 8502 --server.fileWatcherType none &" +
#           "streamlit run seg_predict_app.py --server.port 8602 --server.fileWatcherType none &" +
#           "streamlit run det_predict_app.py --server.port 8702 --server.fileWatcherType none &"
#           )

os.system("streamlit run det_predict_app.py --server.port 8702 --server.fileWatcherType none &"
          )