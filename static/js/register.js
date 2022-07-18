// import axios from "./static/Axios/Axios@0.25.0.min.js";

function bind_captcha_onclick() {
    let captcha_button = document.getElementById("captcha-btn");
    captcha_button.onclick = function () {
        const email = document.getElementsByName("email")[0].value;
        if (!email) {
            alert("请先输入邮箱");
            return;
        } else {
            axios({
                method: "post",
                url: "get_captcha",
                data: {
                    "email": email
                }
            }).then(res => {
                captcha_button.onclick = null;
                let countdown = 60;
                let timer = setInterval(function () {
                    if (countdown > 0) {
                        captcha_button.innerText = countdown + "s";
                        captcha_button.disable = true;
                    } else {
                        captcha_button.innerText = "获取验证码";
                        captcha_button.disable = false;
                        bind_captcha_onclick();
                        clearInterval(timer);
                    }
                    countdown -= 1;
                }, 1000);
                console.log(res);
            })
        }
    };
}


window.onload = function () {
    bind_captcha_onclick();
}