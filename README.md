# О проекте

Простая модель для распознавания цифр из NNIST

## как тренировать
Для тренировки нужно запустить сервер mlflow

    mlflow ui -p 8888

Порт выбирается пользователем, но его нужно будет указать в конфиге. По дефолту -p 8888. Приведу кусок конфига, который хранит эту информацию:

    loggers:
        mlflow:
            experiment_name: first_exp
            tracking_uri: http://localhost:8888

Далее просто запускаете

    python MLOps_project/train.py

параметры для тренировки находятся в конфиге. При первом запуске при отсутствии данных они подгрузятся через dvc. Это можно найти в `MLOps_project/data.py`

# MLFlow
## инференс

Чтобы запустить сервер

    python MLOps_project/run_server.py --port=8890

порт выбираете сами. После запуска инференса выдастся адрес, по которой можно работать с моделью:

    Listening at: http://127.0.0.1:8890 (76870)

С этим адресом нужно запустить server test

    python MLOps_project/server_test.py --servind_addr=http://127.0.0.1:8890

serving_addr тот который у вас. На localhost может поменяться порт. /invocations добавляется само, так что это не нужно делать при запуске.


# Trython

## выбор параметров

рассмотрел параметры
```
    instance_group.count
    dynamic_batching.max_queue_delay_microseconds
    dynamic_batching.preferred_batch_size
```

Так как у меня нет GPU и возможно серверу выделить только 5 ядер, instance_group.count = 1 оказался более предпочтительным. Также max_queue_delay = 2000 показал меньшую задержку по сравнению с 1000, так что взял их.


- MacOS 14.1.1 (23B81)
- Apple M1 Pro
- ..
- решается задача распознавания цифр в мнисте, модель принимает картинки и возвращает вектор из 10 координат
- model_repository:



```
    model_repository
    └── onnx-model
        ├── 1
        │   └── model.onnx
        └── config.pbtxt
```

- метрики throughput и latency

| count | max_queue_delay | throughput (infer/sec)| latency (usec)|
|-------|-----------------|------------|---------|
|   1   |       1000      | 8893.17      |   2248      |
|   1   |       2000      | 10556.8    |   1894    |
|    2   |         1000        |      9727.4      |     2055    |
|    2   |        2000         |      10066.7      |     1985    |

## конвертация модели
    python MLOps_project/convert_model.py ./models/best_model.xyz ./model_repository/onnx-model/1/model.onnx

## server_client

Для первого запуска нужно загрузить модель. Далее запуск сервера

    dvc pull model_repository/onnx-model/1/model.onnx.dvc 
    docker compose up

запуск клиента делается максимально просто

    python MLOps_project/client_triton.py

## важное

если хотите прервать тренировку или сервер, то используйте `cmd + C`, это завершит все процессы задействованные в работе программы.

## команды всякие

    docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk

perf_analyzer

    perf_analyzer -m onnx-model -u localhost:8500 --concurrency-range 20:20 --shape IMAGES:2,1,28,28
