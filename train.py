from algo.agent import CustomAgent
from algo.utils import display_train_result, read_stock_data, indicator
from algo.methods import train_custom_model, evaluate_custom_model

def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False):
    custom_agent = CustomAgent(state_size=window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)

    train_data = read_stock_data(train_stock)
    val_data = read_stock_data(val_stock)

    initial_offset = val_data['Adj Close'][1] - val_data['Adj Close'][0]
    indi_train = indicator(train_data)
    indi_val = indicator(val_data)
    for episode in range(1, ep_count + 1):
        train_result = train_custom_model(custom_agent, episode, train_data, indicator=indi_train, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_custom_model(custom_agent, val_data, indicator=indi_val, window_size=window_size, debug=debug)
        display_train_result(train_result, val_result, initial_offset)


if __name__ == "__main__":

    train_stock = '/content/Algorithmic_Trading/data/RELIANCE.csv'
    val_stock = '/content/Algorithmic_Trading/data/RELIANCE_2023.csv'
    window_size = 12
    batch_size = 128
    ep_count = 15
    strategy = "double-dqn"
    model_name = "./model/model_RELIANCE_double_dqn"
    pretrained = False
    debug = False


    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name,
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
