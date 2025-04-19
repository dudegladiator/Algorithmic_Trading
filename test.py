# Testing
import os
from algo.agent import CustomAgent
from algo.utils import display_eval_result, read_stock_data, indicator
from algo.methods import evaluate_custom_model

def main(eval_stock, window_size, model_name, debug):

    data = read_stock_data(eval_stock)
    initial_offset = data['Adj Close'][1] - data['Adj Close'][0]
    indi = indicator(data)

    # Single Model Evaluation
    if model_name is not None:
        custom_agent = CustomAgent(window_size, pretrained=True, model_name=model_name)
        profit, _ = evaluate_custom_model(custom_agent, data, indi, window_size, debug)
        display_eval_result(model_name, profit, initial_offset)

    # Multiple Model Evaluation
    else:
        for model in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model)):
                custom_agent = CustomAgent(window_size, pretrained=True, model_name=model)
                profit = evaluate_custom_model(custom_agent, data, window_size, debug)
                display_eval_result(model, profit, initial_offset)
                del custom_agent


if __name__ == "__main__":

    eval_stock = '/content/Algorithmic_Trading/data/RELIANCE_2023.csv'
    window_size = 12
    model_name = '/content/RELIANCE_double_dqn_12_2.h5'
    debug = True
    try:
        main(eval_stock, window_size, model_name, debug)
    except KeyboardInterrupt:
        print("Aborted")
