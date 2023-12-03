Чтобы запустить сервер
    python MLOps_project/run_server.py --port=8890

порт выбираете сами. После запуска инференса выдастся

    Listening at: http://127.0.0.1:8890 (76870)

аддрес в который нужно отправлять запросы. С ним нужно запустить server test

    python MLOps_project/server_test.py --servind_addr=http://127.0.0.1:8890

serving_addr тот который у вас. На localhost может поменяться порт. /invocations добавляется само, так что это не нужно делать при запуске.
