import copy
from models.lstm_model import FedVocLSTMModel


class LSTMServer:
    """
    FedAvg aggregation server for LSTM model.
    Mirrors server_base.py but uses FedVocLSTMModel.
    Weighted aggregation by dataset size — same fix as transformer server.
    """

    def __init__(self, vocab_size, d_model=128, num_layers=1, dropout=0.3, rank=16):
        self.global_model = FedVocLSTMModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            rank=rank,
        )

    def aggregate(self, client_weights_list, dataset_weights=None):
        n = len(client_weights_list)
        if dataset_weights is None:
            dataset_weights = [1.0 / n] * n

        new_state = copy.deepcopy(self.global_model.state_dict())
        for key in new_state.keys():
            new_state[key] = sum(
                w * weights[key]
                for w, weights in zip(dataset_weights, client_weights_list)
            )
        self.global_model.load_state_dict(new_state)