$(document).ready(function () {
    $("#layers_submit").click(function () {
        let modelData = {};
        let modelName;
        const forms = $("form");
        forms.on("submit", function (event) {
            event.preventDefault();
            let data = {};
            let form = new FormData(this);
            for (key of form.keys()) {
                data[key] = form.get(key);
            }
            if (this.name === "model_name") {
                modelName = data["model_name"];
            } else {
                modelData[this.name] = data;
            }

        })
        forms.submit();
        modelData = JSON.stringify(modelData);
        $.ajax({
            xhr: function () {
                return new XMLHttpRequest();
            },
            type: "POST",
            url: "/models/save_model",
            cache: false,
            data: {
                "model_name": modelName,
                "model": modelData
            },
            // dataType: "json",
            contentType: "application/x-www-form-urlencoded"
        }).done(function (res) {
            //    TODO 增加成功信息和错误信息
        }).fail(function (res) {

        })
    });

})