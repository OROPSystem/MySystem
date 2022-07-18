const format1 = 'my-images/              my-images/\n' +
    '├── cat/                ├── train_orgin/\n' +
    '│   ├── 1.jpg           │   ├── cat/\n' +
    '│   └── 2.jpg           │   │   └── *.jpg\n' +
    '└── dog/                │   └── dog/\n' +
    '    ├── 1.jpg           │       └── *.jpg\n' +
    '    └── 2.jpg           └── test/\n' +
    '                            ├── cat/\n' +
    '                            │   └── *.jpg\n' +
    '                            └── dog/\n' +
    '                                └── *.jpg';

const format2 = 'my-images/               my-images/             my-images/\n' +
    '├── 1.jpg                ├── images/             ├── 1.jpg\n' +
    '├── 2.jpg                │   ├── 1.jpg           ├── 2.jpg\n' +
    '├── 3.jpg                │   └── 2.jpg           ├── 3.jpg\n' +
    '└── labels.txt           └── labels.txt          ├── train_orgin.txt\n' +
    '                                                 └── test.txt';


const format3 = 'Save your data with:\n\n' +
    'np.savez(filename, x=images, y=labels)\n\n' +
    'or\n\n' +
    'np.savez(filename, x_train=images_train, y_train=labels_train, x_test=images_test,   y_test=labels_test)';

const formats = {
    'option1': format1,
    'option2': format2,
    'option3': format3
};


const compressed_file = ".zip, .rar, .tar"

$(document).ready(function () {
    $("#image_file").on("change", function () {
        $(".progress").addClass("invisible")
        $("#upload_form_button").attr("disabled", false)
        show_filename("image_file_label", this.files)
    }).attr("accept", compressed_file)

    //Format Select
    $("#format-content").text(formats["option1"])
    $("#select_option").on("change", function () {
        change_pre($("#select_option").find(":selected").text())
    })


    $("#upload_form").on("submit", function (event) {
        event.preventDefault();
        let form_data = new FormData($("#upload_form")[0]);
        $("#success_info").addClass("hidden").html('');
        $("#image_file_error").addClass("hidden").html('');
        $(".progress").removeClass("invisible")
        $.ajax({
            xhr: function () {
                let xhr = new XMLHttpRequest();
                xhr.upload.addEventListener("progress", function (e) {
                    if (e.lengthComputable) {
                        let percent = Math.round(e.loaded / e.total * 100);
                        $(".progress-bar").attr('aria-valuenow', percent).css('width', percent + '%').text(percent + '%')
                    }
                });
                return xhr;
            },
            type: "POST",
            url: '/datasets/save_image',
            cache: false,
            data: form_data,
            processData: false,
            contentType: false
        }).done(function (res) {
            if (res['code'] === 200) {
                $("#success_info").removeClass("hidden").html(res["message"]);
            } else {
                if (res["image_file"]) {
                    $("#image_file_error").removeClass("hidden").html(res["image_file"][0])
                }
            }
        }).fail(function (res) {
            console.log("POST失败")
        })
    })

    function show_filename(id, files) {
        $("#" + id).html(files[0].name)
    }

    function change_pre(option) {
        if (option === "Folder per class") {
            $("#image_file_type b").html("Image compression file")
            $("#image_file").attr("accept", compressed_file)
            $("#format-content").text(formats["option1"]);
        } else if (option === "All same folder with label file") {
            $("#image_file_type b").html("Image compression file")
            $("#image_file").attr("accept", compressed_file)
            $("#format-content").text(formats["option2"]);
        } else if (option === "Numpy file") {
            $("#image_file_type b").html("Numpy file")
            $("#image_file").attr("accept", ".npz")
            $("#format-content").text(formats["option3"]);
        }
    }
})