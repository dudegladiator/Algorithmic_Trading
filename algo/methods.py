import logging
import numpy as np
from tqdm import tqdm

from .utils import (
    format_curr,
    format_pos,
    get_state_representation,
)

logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG
def train_custom_model(agent, episode, df, indicator, ep_count=20, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(df) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state_representation(df, 0, window_size + 1, indicator)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc=f'Episode {episode}/{ep_count}'):
        reward = 0
        next_state = get_state_representation(df, t + 1, window_size + 1, indicator)

        action = agent.take_action(state)

        if action == 1:
            agent.inventory.append(df['Adj Close'][t])

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = df['Adj Close'][t] - bought_price
            reward = delta
            total_profit += delta

        else:
            pass

        done = (t == data_length - 1)
        agent.store_memory(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    agent.save_model(episode)

    return episode, ep_count, total_profit, np.mean(np.array(avg_loss))


def evaluate_custom_model(agent, df, indicator, window_size, debug):
    total_profit = 0
    data_length = len(df) - 1

    history = []
    agent.inventory = []
    data = list(df['Adj Close'])
    state = get_state_representation(df, 0, window_size + 1, indicator)

    for t in range(data_length):
        reward = 0
        next_state = get_state_representation(df, t + 1, window_size + 1, indicator)

        action = agent.take_action(state, is_eval=True)

        if action == 1:
            agent.inventory.append(data[t])
            # agent.inventory.append((df['Adj Close'][t],t))
            history.append((t, "BUY"))
            if debug:
                print("Buy at: {}".format(format_curr(data[t])))
                # logging.debug("Buy at: {}".format(format_curr(df['Adj Close'][t])))

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            # bought_price,t = agent.inventory.pop(0)
            delta = df['Adj Close'][t] - bought_price
            reward = delta
            total_profit += delta
            history.append((t, "SELL"))
            if debug:
                print("Sell at: {} | Position: {}".format(format_curr(data[t]), format_pos(data[t] - bought_price)))
                # logging.debug("Sell at: {} | Position: {}".format(
                #     format_curr(df['Adj Close'][t]), format_pos(df['Adj Close'][t] - bought_price)))

        else:
            history.append((t, "HOLD"))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            for i in range(len(agent.inventory)):
                total_profit += df['Adj Close'][t] - agent.inventory.pop(0)
            print('Profit:  {}'.format(format_pos(total_profit)))
            return total_profit, history