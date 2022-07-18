$(document).ready(function () {
    $("#train_file").on("change", function () {
        $(".progress").addClass("invisible")
        $("#upload_form_button").attr("disabled", false)
        show_filename("train_file_label", this.files)
    })

    $("#test_file").change(function () {
        $(".progress").addClass("invisible")
        show_filename("test_file_label", this.files)
    })

    $("#upload_form").on("submit", function (event) {
        event.preventDefault();
        let form_data = new FormData($("#upload_form")[0]);
        $("#success_info").addClass("hidden").html('');
        $("#train_file_error").addClass("hidden").html('');
        $("#test_file_error").addClass("hidden").html('');
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
            url: '/datasets/save_tabular',
            cache: false,
            data: form_data,
            processData: false,
            contentType: false
        }).done(function (res) {
            if (res['code'] === 200) {
                $("#success_info").removeClass("hidden").html(res["message"]);
            } else {
                if (res["train_file"]) {
                    $("#train_file_error").removeClass("hidden").html(res["train_file"][0])
                }
                if (res["test_file"]) {
                    $("#test_file_error").removeClass("hidden").html(res["test_file"][0])
                }
            }
        }).fail(function (res) {
            console.log("POST失败")
        })
    })

    function show_filename(id, files) {
        $("#" + id).html(files[0].name)
    }
})
