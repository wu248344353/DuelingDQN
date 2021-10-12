import copy
import numpy as np
from N_step_predict import *

direction_to_action = {"up": [2, 1, 3], "right": [1, 3, 0], "down": [3, 0, 2], "left": [0, 2, 1]}
action_to_direction = {0: "down", 1: "up", 2: "left", 3: "right"}
predict_d_position = [[-1, 0], [1, 0], [0, -1], [0, 1]]


# Network输出动作转换为环境接收的动作
def output_to_action(output, direction):
    return direction_to_action[direction][output]


# 贪吃蛇的运动方向
def obs_direction(observation):
    snake_direction = "up"
    state = copy.deepcopy(observation)
    snake_idx = state["controlled_snake_index"]
    snake = state[snake_idx]
    snake_head = snake[0]  # 头的坐标
    snake_tail = snake[1]
    height, width = state["board_height"], state["board_width"]
    if (snake_head[0] == ((snake_tail[0] + 1) % height)) and (snake_head[1] == snake_tail[1]):  # up
        snake_direction = "up"
    elif (snake_head[0] == ((snake_tail[0] + height - 1) % height)) and (snake_head[1] == snake_tail[1]):  # down
        snake_direction = "down"
    elif (snake_head[0] == snake_tail[0]) and (snake_head[1] == ((snake_tail[1] + 1) % width)):  # right
        snake_direction = "right"
    elif (snake_head[0] == snake_tail[0]) and (snake_head[1] == ((snake_tail[1] + width - 1) % width)):  # left
        snake_direction = "left"
    return snake_direction


# 返回蛇头位置，食物位置，危险位置
def obs_position(observation):
    state = copy.deepcopy(observation)
    snake_idx = state["controlled_snake_index"]
    snake_position = state[snake_idx]
    food_position = state[1]
    danger_position = []
    for idx in state.keys():
        if (idx != 1) and (idx != snake_idx) and isinstance(idx, int):
            head = 0
            competitor = state[idx]
            for coordinate in competitor:
                if head == 0:
                    for d_coor in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                        head_danger = [(coordinate[0] + d_coor[0] + 10) % 10, (coordinate[1] + d_coor[1] + 20) % 20]
                        if head_danger not in danger_position:
                            danger_position.append(head_danger)
                    head = 1
                if coordinate not in danger_position:
                    danger_position.append(coordinate)
    return snake_position, food_position, danger_position


# 预测动作是否合理
def predict_actions(snake, food, danger, direction):
    all_actions = copy.deepcopy(STEP_3)
    alive_actions = copy.deepcopy(STEP_3)
    good_actions, scores = [], []
    for actions in all_actions:
        predict_foods = copy.deepcopy(food)
        predict_snake = copy.deepcopy(snake)
        predict_danger = copy.deepcopy(danger) + predict_snake[1:]
        predict_direction = copy.deepcopy(direction)
        score = 0
        for action in actions:
            move_action = direction_to_action[predict_direction][action]  # 0-3的动作
            nearly_postion = copy.deepcopy(predict_d_position[move_action])
            predict_head = [(predict_snake[0][0] + nearly_postion[0]) % 10,
                            (predict_snake[0][1] + nearly_postion[1]) % 20]
            predict_direction = copy.deepcopy(action_to_direction[move_action])
            if predict_head in predict_danger:
                alive_actions.remove(actions)
                if actions in good_actions:
                    good_actions.remove(actions)
                break
            elif predict_head in predict_foods:
                if actions not in good_actions:
                    good_actions.append(copy.deepcopy(actions))
                predict_snake.insert(0, copy.deepcopy(predict_head))
                score += 1
            else:
                predict_snake.insert(0, copy.deepcopy(predict_head))
                predict_snake.pop()  # 尾巴出列
        if actions in good_actions:
            scores.append(copy.deepcopy(score))
    if alive_actions == []:
        all_actions = copy.deepcopy(STEP_2)
        alive_actions = copy.deepcopy(STEP_2)
        good_actions, scores = [], []
        for actions in all_actions:
            predict_foods = copy.deepcopy(food)
            predict_snake = copy.deepcopy(snake)
            predict_danger = copy.deepcopy(danger) + predict_snake[1:]
            predict_direction = copy.deepcopy(direction)
            score = 0
            for action in actions:
                move_action = direction_to_action[predict_direction][action]  # 0-3的动作
                nearly_postion = copy.deepcopy(predict_d_position[move_action])
                predict_head = [(predict_snake[0][0] + nearly_postion[0]) % 10,
                                (predict_snake[0][1] + nearly_postion[1]) % 20]
                predict_direction = copy.deepcopy(action_to_direction[move_action])
                if predict_head in predict_danger:
                    alive_actions.remove(actions)
                    if actions in good_actions:
                        good_actions.remove(actions)
                    break
                elif predict_head in predict_foods:
                    if actions not in good_actions:
                        good_actions.append(copy.deepcopy(actions))
                    predict_snake.insert(0, copy.deepcopy(predict_head))
                    score += 1
                else:
                    predict_snake.insert(0, copy.deepcopy(predict_head))
                    predict_snake.pop()  # 尾巴出列
            if actions in good_actions:
                scores.append(copy.deepcopy(score))
    if alive_actions == []:
        all_actions = copy.deepcopy(STEP_1)
        alive_actions = copy.deepcopy(STEP_1)
        good_actions, scores = [], []
        for actions in all_actions:
            predict_foods = copy.deepcopy(food)
            predict_snake = copy.deepcopy(snake)
            predict_danger = copy.deepcopy(danger) + predict_snake[1:]
            predict_direction = copy.deepcopy(direction)
            score = 0
            for action in actions:
                move_action = direction_to_action[predict_direction][action]  # 0-3的动作
                nearly_postion = copy.deepcopy(predict_d_position[move_action])
                predict_head = [(predict_snake[0][0] + nearly_postion[0]) % 10,
                                (predict_snake[0][1] + nearly_postion[1]) % 20]
                predict_direction = copy.deepcopy(action_to_direction[move_action])
                if predict_head in predict_danger:
                    alive_actions.remove(actions)
                    if actions in good_actions:
                        good_actions.remove(actions)
                    break
                elif predict_head in predict_foods:
                    if actions not in good_actions:
                        good_actions.append(copy.deepcopy(actions))
                    predict_snake.insert(0, copy.deepcopy(predict_head))
                    score += 1
                else:
                    predict_snake.insert(0, copy.deepcopy(predict_head))
                    predict_snake.pop()  # 尾巴出列
            if actions in good_actions:
                scores.append(copy.deepcopy(score))
    choose_action = None
    if len(scores) > 0:
        max_idx = 0
        for i in range(len(scores)):
            if scores[i] > scores[max_idx]:
                max_idx = copy.deepcopy(i)
        choose_action = good_actions[max_idx][0]
    # 返回所有可用的动作和分数最高的动作
    return alive_actions, choose_action


# 观测信息转置
def transform(feature, direction):
    fea = feature
    if direction == "up":
        fea = feature
    elif direction == "right":
        # 逆时针旋转90度
        fea = np.transpose(feature)
        fea = fea[::-1]
    elif direction == "down":
        # 逆时针旋转180度
        fea = np.transpose(feature)
        fea = fea[::-1]
        fea = np.transpose(fea)
        fea = fea[::-1]
    elif direction == "left":
        # 逆时针旋转270度
        fea = np.transpose(feature)
        fea = fea[::-1]
        fea = np.transpose(fea)
        fea = fea[::-1]
        fea = np.transpose(fea)
        fea = fea[::-1]
    return fea


# 观测信息提取
def head_and_obs(state, idx):
    snake = state[idx]
    head = snake[0]
    tail = snake[1]
    height, width = state["board_height"], state["board_width"]
    direction = "up"
    if (head[0] == ((tail[0] + 1) % height)) and (head[1] == tail[1]):  # up
        direction = "up"
    elif (head[0] == ((tail[0] + height - 1) % height)) and (head[1] == tail[1]):  # down
        direction = "down"
    elif (head[0] == tail[0]) and (head[1] == ((tail[1] + 1) % width)):  # right
        direction = "right"
    elif (head[0] == tail[0]) and (head[1] == ((tail[1] + width - 1) % width)):  # left
        direction = "left"
    obs = state
    feature = []
    agent = idx
    fea_screen = np.zeros((height, width))  # 特征0：食物是1，身体是-1
    perfect_feature = np.zeros((height, height))  # 特征0：食物是1，身体是-1

    snake_head = []
    snake_length = 0
    for num in obs.keys():
        if num == 1:
            data = obs[num]
            for coordinate in data:
                fea_screen[coordinate[0], coordinate[1]] = 1
        elif num == agent:  # 自己的信息
            data = obs[num]
            snake_head = data[0]
            snake_length = [len(data)]
            for coordinate in data:
                fea_screen[coordinate[0], coordinate[1]] = -1
        elif isinstance(num, int):
            data = obs[num]
            head = 0
            for coordinate in data:
                if head == 0:
                    fea_screen[coordinate[0], coordinate[1]] = -2
                    head = 1
                else:
                    fea_screen[coordinate[0], coordinate[1]] = -1
    for x in range(height):
        for y in range(height):
            dx = (snake_head[0] + height + (x - 5)) % height
            dy = (snake_head[1] + width + (y - 5)) % width
            perfect_feature[x, y] = fea_screen[dx, dy]
    perfect_feature = transform(perfect_feature, direction)
    perfect_feature = perfect_feature.reshape(-1).tolist()
    feature = perfect_feature[:24] + snake_length + perfect_feature[25:]
    # feature.append(perfect_feature + snake_length)
    # feature.append(perfect_feature)
    return direction, feature


# 判断贪吃蛇是否死亡
def Judge_done(state, next_state):
    my_control = state["controlled_snake_index"]
    DONE = 0
    snake_length = len(state[my_control])
    next_snake_length = len(next_state[my_control])
    if next_snake_length < snake_length:
        DONE = 1
    return DONE


# 贪吃蛇特别长时采用防御策略
def defense(state):
    # 所控制的agent长度大于length，就进入防御状态
    action = None
    defense_length = 25
    agent_idx = state["controlled_snake_index"]
    snake_length = len(state[agent_idx])
    if snake_length >= defense_length:
        # 进入防御状态，环绕而行。
        snake = state[agent_idx]
        height, width = state["board_height"], state["board_width"]
        head = snake[0]  # 蛇头
        tail = snake[-1]  # 蛇尾
        if ((head[0] == ((tail[0] + 1) % height)) and (head[1] == tail[1])):  # 尾巴在蛇头下边
            action = 0
        elif ((head[0] == ((tail[0] + height - 1) % height)) and (head[1] == tail[1])):  # 尾巴在蛇头上边
            action = 1
        elif (head[0] == tail[0]) and (head[1] == ((tail[1] + 1) % width)):  # 尾巴在蛇头左边
            action = 2
        elif (head[0] == tail[0]) and (head[1] == ((tail[1] + width - 1) % width)):  # 尾巴在蛇头右边
            action = 3
    return action

