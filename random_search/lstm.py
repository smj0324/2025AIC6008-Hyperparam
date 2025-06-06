from random_search.random_choices import get_random_search_choices
from test.lstm import test_lstm
import random

# LSTM용 하이퍼파라미터 후보 범위 정의
default_ranges = {
    "epochs":            [3, 5, 7, 10],
    "batch_size":        [16, 32, 64],
    "learning_rate":     [0.0001, 0.0005, 0.001, 0.005],
    "units":             [32, 64, 128],
    "dropout":           [0.0, 0.2, 0.5],
    "recurrent_dropout": [0.0, 0.2, 0.5],
    "num_layers":        [1, 2],
    "bidirectional":     [True, False],
    "dense_dropout":     [0.0, 0.3, 0.5]
}

def random_param_choices_lstm(user_hyperparams):
    """
    user_hyperparams에 기반해, 기본 후보 범위(default_ranges)에서
    get_random_search_choices를 이용해 랜덤 탐색 후보군을 생성.
    """
    param_choices = {
        "epochs":            get_random_search_choices(
                                 user_hyperparams.get("epochs"),
                                 default_ranges["epochs"], around=2
                             ),
        "batch_size":        get_random_search_choices(
                                 user_hyperparams.get("batch_size"),
                                 default_ranges["batch_size"], around=16
                             ),
        "learning_rate":     get_random_search_choices(
                                 user_hyperparams.get("learning_rate"),
                                 default_ranges["learning_rate"], scale=10
                             ),
        "units":             get_random_search_choices(
                                 user_hyperparams.get("units"),
                                 default_ranges["units"], scale=2
                             ),
        "dropout":           get_random_search_choices(
                                 user_hyperparams.get("dropout"),
                                 default_ranges["dropout"], scale=1
                             ),
        "recurrent_dropout": get_random_search_choices(
                                 user_hyperparams.get("recurrent_dropout"),
                                 default_ranges["recurrent_dropout"], scale=1
                             ),
        "num_layers":        get_random_search_choices(
                                 user_hyperparams.get("num_layers"),
                                 default_ranges["num_layers"], scale=1
                             ),
        "bidirectional":     get_random_search_choices(
                                 user_hyperparams.get("bidirectional"),
                                 default_ranges["bidirectional"]
                             ),
        "dense_dropout":     get_random_search_choices(
                                 user_hyperparams.get("dense_dropout"),
                                 default_ranges["dense_dropout"], scale=1
                             )
    }
    print("[DEBUG] LSTM param_choices:", param_choices)
    return param_choices

def start_lstm_random_search(user_parameters):
    """
    사용자 정의 하이퍼파라미터(user_parameters)를 기반으로
    랜덤 서치를 수행하여 최종 LSTM 모델을 생성·반환.
    """
    # 1) 후보군 생성
    param_choices = random_param_choices_lstm(user_parameters)

    # 2) 후보군 중 하나씩 무작위 선택
    units             = random.choice(param_choices["units"])
    dropout           = random.choice(param_choices["dropout"])
    recurrent_dropout = random.choice(param_choices["recurrent_dropout"])
    num_layers        = random.choice(param_choices["num_layers"])
    bidirectional     = random.choice(param_choices["bidirectional"])
    dense_dropout     = random.choice(param_choices["dense_dropout"])
    epochs            = random.choice(param_choices["epochs"])
    batch_size        = random.choice(param_choices["batch_size"])
    learning_rate     = random.choice(param_choices["learning_rate"])

    print(
        f"선택된 파라미터: epochs={epochs}, batch_size={batch_size}, "
        f"learning_rate={learning_rate}, units={units}, dropout={dropout}, "
        f"recurrent_dropout={recurrent_dropout}, num_layers={num_layers}, "
        f"bidirectional={bidirectional}, dense_dropout={dense_dropout}"
    )

    # 3) user_parameters 복사 후 LSTM 전용 값으로 업데이트
    hyperparameters = user_parameters.copy()
    hyperparameters.update({
        # IMDB 용 기본값(예: vocab_size, sequence_length, embedding_dim)은
        # test_lstm() 내부에서 이미 하드코딩되어 있으므로, 이곳엔
        # 오직 랜덤으로 선택된 LSTM 관련 파라미터만 넣습니다.
        "units":             units,
        "dropout":           dropout,
        "recurrent_dropout": recurrent_dropout,
        "num_layers":        num_layers,
        "bidirectional":     bidirectional,
        "dense_dropout":     dense_dropout
    })

    # 4) test_lstm() 호출 (인자 없이 그대로 실행)
    model, X_train, y_train, training_hyperparams = test_lstm(hyperparameters)

    # 5) test_lstm()이 반환한 training_hyperparams에 epochs, batch_size, learning_rate 적용
    training_hyperparams.update({
        "epochs":        epochs,
        "batch_size":    batch_size,
        "learning_rate": learning_rate
    })
    print("[DEBUG] 최종 training_hyperparams:", training_hyperparams)
    
    return model, X_train, y_train, training_hyperparams
