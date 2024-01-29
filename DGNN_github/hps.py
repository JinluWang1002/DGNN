# epoch：1000 patience：25
hp_cora = {
    "wd2": 2e-4,
    "wdy": 4e-5,
    "lambda_1": 1.0,
    "lambda_2": 2.0,
    "lambda_3": 0.02,
    "epsilon": 0.5,
    "layer": 2,
    "dropout": 0.25,
    "lr": 0.002,
}


# epoch：1000 patience：25
hp_citeseer = {
    "wd2": 2e-4,
    "wdy": 5e-2,
    "lambda_1": 2.0,
    "lambda_2": 0.5,
    "lambda_3": 0.01,
    "epsilon": 0.5,
    "layer": 3,
    "dropout": 0.15,
    "lr": 0.003,
}

# epoch：1000 patience：25
hp_chameleon = {
    "wd2": 2e-6,
    "wdy": 2e-8,
    "lambda_1": 1.0,
    "lambda_2": 2.5,
    "lambda_3": 1e-2,
    "epsilon": 0.5,
    "layer": 2,
    "dropout": 2e-2,
    "lr": 0.05,
}


# epoch：1000 patience：25
hp_squirrel = {
    "wd2": 0.0,
    "wdy": 2e-8,
    "lambda_1": 1.0,
    "lambda_2": 2.5,
    "lambda_3": 0.01,
    "epsilon": 0.5,
    "layer": 2,
    "dropout": 0.0,
    "lr": 0.02,
}

# epoch：1000 patience：25
hp_amazon_computers = {
    "wd2": 0,
    "wdy": 0,
    "lambda_1": 2.0,
    "lambda_2": 1.0,
    "lambda_3": 0.01,
    "epsilon": 0.2,
    "layer": 2,
    "dropout": 0.05,
    "lr": 0.03,
}

# set_seed(3407)
# epoch：1000 patience：25
hp_amazon_photo = {
    "wd2": 0,
    "wdy": 5e-8,
    "lambda_1": 2.0,
    "lambda_2": 0.5,
    "lambda_3": 1e-2,
    "epsilon": 0.5,
    "layer": 2,
    "dropout": 0.15,
    "lr": 0.02
}


def get_hyper_param(name: str):
    name = name.lower()
    if name == "cora":
        return hp_cora
    elif name == "citeseer":
        return hp_citeseer
    elif name == "chameleon":
        return hp_chameleon
    elif name == "squirrel":
        return hp_squirrel
    elif name == "computers":
        return hp_amazon_computers
    elif name == "photo":
        return hp_amazon_photo
    else:
        raise Exception("Not available")
