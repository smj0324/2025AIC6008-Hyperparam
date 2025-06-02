from random_search.random_choices import get_random_search_choices
from test.mobilenet import test_moblinet
import random

default_ranges = {
    "epochs":      [3, 5, 7, 10],
    "batch_size":  [16, 32, 64],
    "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
    "alpha":       [0.75, 1.0, 1.25],
    "minimalistic": [True, False]
}

def random_param_choices(user_hyperparams):
    param_choices = {
        "epochs": get_random_search_choices(user_hyperparams.get("epochs"), default_ranges["epochs"], around=2),
        "batch_size": get_random_search_choices(user_hyperparams.get("batch_size"), default_ranges["batch_size"], around=16),
        "learning_rate": get_random_search_choices(user_hyperparams.get("learning_rate"), default_ranges["learning_rate"], scale=10),
        "alpha": get_random_search_choices(user_hyperparams.get("alpha"), default_ranges["alpha"], scale=1.25),
        "minimalistic": get_random_search_choices(user_hyperparams.get("minimalistic"), default_ranges["minimalistic"])
    }
    print("[DEBUG] param_choices:", param_choices)
    return param_choices

def filter_alpha_by_minimalistic(alpha_choices, minimalistic, weights):
    """minimalistic과 weights에 따라 alpha 후보군 필터링"""
    if minimalistic:
        allowed = [1.0]
    else:
        allowed = [0.75, 1.0]

    if weights is None:
        return alpha_choices
    else:
        return [a for a in alpha_choices if a in allowed]

def start_moblinet_random_search(user_parameters):
    param_choices = random_param_choices(user_parameters)
    minimalistic = random.choice(param_choices["minimalistic"])
    
    weights = user_parameters.get("weights", "imagenet")

    filtered_alpha_choices = filter_alpha_by_minimalistic(param_choices["alpha"], minimalistic, weights if weights != "None" else None)
    
    alpha = random.choice(filtered_alpha_choices)

    if not filtered_alpha_choices:
        alpha = 1.0

    epochs = random.choice(param_choices["epochs"])
    batch_size = random.choice(param_choices["batch_size"])
    learning_rate = random.choice(param_choices["learning_rate"])

    if (minimalistic and alpha != 1.0) or (not minimalistic and alpha not in [0.75, 1.0]):
        weights = None

    print(f"선택된 파라미터: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, alpha={alpha}, minimalistic={minimalistic}, weights={weights}")

    hyperparameters = user_parameters.copy()
    hyperparameters["alpha"] = alpha
    hyperparameters["minimalistic"] = minimalistic
    hyperparameters["weights"] = weights

    model, X_train, y_train, training_hyperparams = test_moblinet(
        mv_base_model_params=hyperparameters,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    return model, X_train, y_train, training_hyperparams
