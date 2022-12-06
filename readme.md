# 缺陷智能检测系统2.0

在2.0版本中，我们首先重构了代码系统框架，采用外部统一接口+多个子接口的设计模式调整了前后端代码架构，并针对子接口添加了缺陷分割、缺陷检测模块，现在的2.0系统可拓展性更强，方便引入更多的功能模块例如异常检测、故障诊断等。

## 缺陷分类数据和模型上传规范

### 数据集规范

1. 命名的唯一性，不能与系统中其他数据集同名
2. 主文件夹下包含用于训练和测试的数据：统一命名为**train_origin**和**test**
3. 训练和测试数据集下包含数据集的每个类别文件夹
4. 将图片以**jpg/png/bmp**格式放置在对应的分类文件夹下，对图片命名无要求，中英文均可
5. 提交压缩包（上限10GB），例如*NEU.zip*，确保解压之后路径结构如下：

**NEU**

- train_origin
  - img_class1
    - img1（jpg/png）
    - img2（jpg/png）
    - img3（jpg/png）
  - img_class2
  - img_class3
- test
  - img_class1
    - img1（jpg/png）
    - img2（jpg/png）
    - img3（jpg/png）
  - img_class2
  - img_class3

### 模型规范

支持加载 **torch** 和 **timm** 库的预训练模型，或者您自定义的模型（没有测试过，如果有需要，欢迎提供自定义模型给我测试并完善一下功能）

- 模型名称

  - 如果使用预训练模型，请使用加载时的模型作为保存的模型名称 + 数据集名称

    例如 `vit_b_16_neu,` `vgg16_pkupcb`，而不是只采用vit作为名称

    参考：[Models and pre-trained weights — Torchvision 0.13 documentation (pytorch.org)](https://pytorch.org/vision/stable/models.html)

- 模型保存：

  - 如果您直接采用预模型的**默认归一化**方式，使用 `torch.save("xxx.pth")`即可

    ```python
    DEFAULT_CROP_PCT = 0.875
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
    IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
    IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)
    ```

  - 如果您引入了自定义的transform，请按照如下格式保存：

    ```python
    net_path = checkpoint_path + 'net.pth' # 名称自定义, 不重名即可
    torch.save({
        "input_size": (3, 224, 224), # define youself
        "model": net, # define youself
        "mean": (0.5, 0.5, 0.5), # define youself
        "std": (0.5, 0.5, 0.5), # define youself
    }, net_path)
    ```

  - 如果您采用保存模型参数（OrderedDict）的方式，请提供一个 `net.py`文件，文件应该包含如下内容：

    ```python
    # 以ResNet为例
    class ResNets(nn.Module):
        def __init__(self, model_name='resnet18', num_layers=12, num_classes=5, feature_extract=False, use_pretrained=True):
            super(ResNets, self).__init__()
    				# 创建模型代码
    
        def forward(self, x):
    				# 前向传播代码
            return result
    
    # 如果前向传播输出的不只是预测的logit, 请说明输出的格式.
    # 例如 [feature, logit] <type `tuple`>
    ```

以上均通过测试，如有问题随时沟通。

优先提供预训练模型。说明预训练模型库，自定义模型类，自己测试达到的精度。

---

## 缺陷分割数据和模型上传规范

### 数据集规范

1. 命名的唯一性，不能与系统中其他数据集同名
2. 文件夹包含imgs和labels两个子文件夹，分别对应原始图像和mask
3. 将图片以**jpg/png/bmp**格式放置在对应的分类文件夹下，对图片命名无要求，中英文均可
4. 提交压缩包（上限10GB）

### 模型规范

支持加载 **segmentation-models-pytorch**  库的预训练模型

- 模型名称

  - 方法名称\_backbone_数据集.pth

    例如 `unet_resnet34_mtd.pth`, `pspnet_resnet50_ksdd.pth` 

- 模型保存：

  - 请直接采用保存权重参数方式，使用 `torch.save("xxx.pth")`即可

    关于，请采用 **segmentation-models-pytorch**  库的默认设置，不要进行修改

    系统会根据如下代码识别模型backbone，自动读取预处理方式：

    ```python
    preprocessing_fn = smp.encoders.get_preprocessing_fn(f"{backbone}", "imagenet")
    preprocessing = get_preprocessing(preprocessing_fn)
    ```

---

## 缺陷检测数据和模型上传规范

### 数据集规范

1. 命名的唯一性，不能与系统中其他数据集同名
2. 文件夹包含JPEGImages子文件夹，对应原始图像，其他json文件等也可以提供
3. 将图片以**jpg/png/bmp**格式放置在对应的分类文件夹下，对图片命名无要求，中英文均可
4. 提交压缩包（上限10GB）

### 模型规范

支持加载 **mmdetection3.0**  库的预训练模型

- 模型名称

  - 方法名称\_backbone_数据集.pth

    例如 `cascade_resnet101_goer.pth`, `frcn_resnet50_xmu.pth` 

- 配置文件

  - 按照`mmdetection`标准的配置文件格式，必须包含`visualizer`，`test_dataloader`，`model`这三个config
  - 命名和模型名称一致，例如`cascade_resnet101_goer.py`，`frcn_resnet50_xmu.py`
  - 配置文件用于inferenc阶段

- 模型保存：

  - 请直接采用保存权重参数方式，使用 `torch.save("xxx.pth")`即可
  - 关于inference阶段的归一化方式，系统会自动识别配置文件中的`test_dataloader`进行图像预处理

---

## 新功能添加步骤

### 1. 添加外部统一接口

1. 在application.html文件中加入card_combine
2. 按钮script脚本

### 2. 添加子界面接口

1. 添加templates_xxx，选择需要的模块html文件
2. 修改base.html，其他html继承base.html，修改继承关系和其他细节部分
3. blueprints中的flask相关部分修改，增加相关功能xxx的页面定向函数
4. 申请一个predict.html的接口，修改相应部分

### 3. 后端代码

1. 新建xxx_predict_app.py
2. 使用streamlit相关脚本实现推断部分
3. 在run_stramlit.py中增加一个接口，并运行xxx_predict_app.py脚本测试

