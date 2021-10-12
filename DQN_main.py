from env.chooseenv import make
from Agent import DQN_Agent
from Functions import *
import random
import torch

MAX_EPISODES = 100000


def rate_of_win(model):
    env = make("snakes_3v3", conf=None)
    win = [0, 0]
    ctrl_agent_index = [2, 3, 4, 5, 6, 7]  # 控制的agent
    pre_model = DQN_Agent(state_dim=100, action_dim=3, isTrain=False)
    for epoch in range(200):
        state = env.reset()
        while True:
            actions = []
            for agent in range(len(ctrl_agent_index)):
                direction, feature = head_and_obs(state[agent], state[agent]["controlled_snake_index"])
                if agent in [0, 1, 2]:
                    out = model.action(np.array(feature))
                else:
                    out = pre_model.action(np.array(feature))
                # out = model.action(np.array(feature))
                snake_position, food_position, danger_position = obs_position(copy.deepcopy(state[agent]))
                all_action, output = predict_actions(snake_position, food_position, danger_position,
                                                     copy.deepcopy(direction))
                False_action = True
                for acts in all_action:
                    if out==acts[0]:
                        False_action = False
                        break
                if False_action == True:
                    if output != None:
                        # print("优势动作")
                        out = output
                    elif len(all_action) > 0:
                        # print("随机动作")
                        out = random.choice(all_action)[0]
                    else:
                        pass
                        # print("无路可走")

                act = output_to_action(out, direction)
                actions.append(act)
            next_state, reward, done, _, info = env.step(env.encode(actions))
            state = next_state
            if done:
                all_score = info["score"]
                if (all_score[0] + all_score[1] + all_score[2]) > (all_score[3] + all_score[4] + all_score[5]):
                    win[0] += 1
                else:
                    win[1] += 1
                break
    if win[0] > win[1]:
        print("胜率：", win[0]/200)
        name = "net_params.pkl"
        model.save_paramaters(name)


def main():
    env = make("snakes_3v3", conf=None)
    num_agents = env.n_player
    width = env.board_width
    height = env.board_height
    ctrl_agent_index = [2, 3, 4, 5, 6, 7]  # 控制的agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DQN_Agent(state_dim=100, action_dim=3, isTrain=True)
    model.save_paramaters("net_params.pkl")
    episode = 0
    score = 0
    while episode < MAX_EPISODES:
        state = env.reset()
        episode += 1
        step = 0
        while True:
            actions, outputs, observations, next_observations, Done = [], [], [], [], []
            for agent in range(len(ctrl_agent_index)):
                # 得到outputs，actions，observations
                direction, feature = head_and_obs(state[agent], state[agent]["controlled_snake_index"])
                out = model.egreedy_action(np.array(feature))
                snake_position, food_position, danger_position = obs_position(copy.deepcopy(state[agent]))
                all_action, output = predict_actions(snake_position, food_position, danger_position, copy.deepcopy(direction))
                False_action = True
                for acts in all_action:
                    if out == acts[0]:
                        False_action = False
                        break
                if False_action == True:
                    if output != None:
                        # print("优势动作")
                        out = output
                    elif len(all_action) > 0:
                        # print("随机动作")
                        out = random.choice(all_action)[0]
                    else:
                        pass
                        # print("无路可走")
                action = output_to_action(out, direction)

                outputs.append(out)
                actions.append(action)
                observations.append(feature)

            next_state, reward, done, _, info = env.step(env.encode(actions))
            for agent in range(len(ctrl_agent_index)):
                direction, feature = head_and_obs(next_state[agent], next_state[agent]["controlled_snake_index"])
                next_observations.append(feature)
                Done.append(Judge_done(state[agent], next_state[agent]))

            for agent in range(len(ctrl_agent_index)):
                # ReplayBuffer
                model.replay_buffer.push(observations[agent], outputs[agent], reward[agent], next_observations[agent], Done[agent])
            if episode > 10:
                # 存储10个episode的经验再进行训练
                model.train()
            state = next_state
            step += 1
            if done:
                obs = state[0]
                for agent_i in {2, 3, 4, 5, 6, 7}:
                    score += len(obs[agent_i])
                if episode % 100 == 0:
                    print("第", episode, "局:", score/100, "分")
                    if score/100 > 20:
                        rate_of_win(model)
                    score = 0
                break


if __name__ == '__main__':
    main()
