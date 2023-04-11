models = {
    1: {
        "model_name": "ResNet34",
        "author": "XuDong Peng",
        "library": "TensorFlow",
        "task": "NEU-CLS",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    2: {
        "model_name": "Vision Transformer",
        "author": "Zongwei Du",
        "library": "PyTorch",
        "task": "Magnetic-Tile-Defect",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    3: {
        "model_name": "Swin Transformer",
        "author": "Liezhi Lu",
        "library": "PyTorch",
        "task": "KTH-TIPS",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    4: {
        "model_name": "DeepLab-v3",
        "author": "Qian Wan",
        "library": "PyTorch",
        "task": "KolektorSDD",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    5: {
        "model_name": "MobileNet",
        "author": "ShenQiang ke",
        "library": "PyTorch",
        "task": "Kylberg-Texture",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    6: {
        "model_name": "ResNet50",
        "author": "Chen Sun",
        "library": "PyTorch",
        "task": "Bridge Crack",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    7: {
        "model_name": "VGG",
        "author": "Yiping Gao",
        "library": "TensorFlow",
        "task": "NEU-CLS",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    8: {
        "model_name": "1D-Resnet18",
        "author": "Shenqiang Ke",
        "library": "Pytorch",
        "task": "HUST-Motor",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    9: {
        "model_name": "WDCNN",
        "author": "Shenqiang Ke",
        "library": "Pytoch",
        "task": "CWRU-Bearing",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    10: {
        "model_name": "LSTM-Attention",
        "author": "Li Wang",
        "library": "Pytoch",
        "task": "PU-Bearing",
        "introduction": "abcdefg hijk lmn opq rst uvw xyz"
    },
    
}
total = 10

function create_model(model_info) {
    let block = $("<div></div>").addClass("model");
    let target = $("<a></a>").attr("href", "javascript:;");
    let model_header = $("<div></div>").addClass("model-header").html(model_info.author + " / " + model_info["model_name"]);
    let model_body = $("<div></div>").addClass("model-info");
    model_body.append("<div>" + model_info["task"] + " / " + "</div>");
    model_body.append("<div>" + " " + model_info["library"] + "</div>");
    target.append(model_header);
    target.append(model_body);
    block.append(target);
    return block
}

function create_intro(model_info) {
    let html = $("<div></div>");
    let header = $("<div></div>").addClass("intro-header").append($("<h3>" + model_info["model_name"] + "</h3>"));
    let introduction = $("<div></div>").addClass("introduction");
    introduction.html(model_info["introduction"]);
    let btn = $("<button>引用</button>").addClass("quote-model");
    html.append(header).append(introduction).append(btn);
    return html
}

$(".model-intro").hide();
$(".hazel").hide().click(function () {
    $(".hazel").hide();
    $(".model-intro").hide();
});

for (let i = 1; i <= total; i++) {
    console.log(i)
    $(".model-container").append(create_model(models[i]));
}

$(".model a").click(function () {
    $(".model-intro").html(create_intro(models[1])).show();
    $(".hazel").show();

    $(".quote-model").click(function () {
        let quote_name = $(".intro-header h3").text();
        $.ajax({
            type: "POST",
            url: "/models_storage/quote_model",
            dataType: "json",
            data: {"quote_name": quote_name},
        }).done(function (res) {
            if (res["code"] === 200) {
                alert("引用完成");
            }
        })
    })
})


