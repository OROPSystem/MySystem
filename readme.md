# 缺陷分类数据和模型上传规范

## 数据集规范

1. 命名的唯一性，不能与系统中其他数据集同名
2. 主文件夹下包含用于训练和测试的数据：统一命名为**train_origin**和**test**
3. 训练和测试数据集下包含数据集的每个类别文件夹
4. 将图片以**jpg/png**格式放置在对应的分类文件夹下，对图片命名无要求
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

------

## 模型规范

支持加载 **torch** 和 **timm** 库的预训练模型，或者您自定义的模型（没有测试过，如果有需要，欢迎提供自定义模型给我测试并完善一下功能）

- 模型名称

  - 如果使用预训练模型，请使用加载时的模型作为保存的模型名称 + 数据集名称

    例如`vit_b_16_neu,` `vgg16_pkupcb`，而不是只采用vit作为名称

    参考：[Models and pre-trained weights — Torchvision 0.13 documentation (pytorch.org)](https://pytorch.org/vision/stable/models.html)

- 模型保存：

  - 如果您直接采用预模型的**默认归一化**方式，使用`torch.save("xxx.pth")`即可

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

以上均通过测试，如有问题随时沟通。

优先提供预训练模型。说明预训练模型库，自定义模型类，自己测试达到的精度。