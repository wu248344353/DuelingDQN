def predict_4_actions():
    predict_actions = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    action = [a, b, c, d]
                    predict_actions.append(action)
    return predict_actions


def predict_3_actions():
    predict_actions = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                action = [a, b, c]
                predict_actions.append(action)
    return predict_actions


def predict_2_actions():
    predict_actions = []
    for a in range(3):
        for b in range(3):
            action = [a, b]
            predict_actions.append(action)
    return predict_actions


def predict_1_actions():
    predict_actions = []
    for a in range(3):
        action = [a]
        predict_actions.append(action)
    return predict_actions


STEP_1 = predict_1_actions()
STEP_2 = predict_2_actions()
STEP_3 = predict_3_actions()
STEP_4 = predict_4_actions()
