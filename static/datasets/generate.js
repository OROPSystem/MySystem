const options = {
    'Classifier - Cluster': 'cluster',
    'Classifier - Decision Tree': 'decision_tree',
    'Regression': 'regression'
};


$(document).ready(function () {
    $("#generate_script").text(appConfig.handle_key.examples["cluster"]);

    $("#select_option").on("change", function () {
        change_script($("#select_option").find(":selected").text());
    })


    $("#dataset_name").on("change", function () {
        if ($("#dataset_name").text() === "") {
            $("#generate_form_button").attr("disabled", false)
        }
    })


    $("#generate_form").on("submit", function (event) {
        event.preventDefault();
        let form_data = new FormData($("#generate_form")[0]);
        console.log(form_data)
        $("#generate_info").addClass("hidden").html('');
        $.ajax({
            type: "POST",
            url: '/datasets/save_generate',
            cache: false,
            data: form_data,
            processData: false,
            contentType: false
        }).done(function (res) {
            if (res['code'] === 200) {
                $("#generate_info").removeClass("hidden").html(res["message"]);
            }
        }).fail(function (res) {
            console.log("POST失败")
        })
    })
});


function change_script(option) {
    if (options[option] === "cluster") {
        $("#generate_script").text(appConfig.handle_key.examples["cluster"]);
    } else if (options[option] === "decision_tree") {
        $("#generate_script").text(appConfig.handle_key.examples["decision_tree"]);
    } else if (options[option] === "regression") {
        $("#generate_script").text(appConfig.handle_key.examples["regression"]);
    }
}