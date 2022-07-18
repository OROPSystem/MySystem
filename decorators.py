from flask import g, redirect, url_for
from functools import wraps


def login_required(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        if hasattr(g, "user"):
            return fun(*args, **kwargs)
        else:
            return redirect(url_for("user.login"))

    return wrapper
