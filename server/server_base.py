import copy
from models.base_model import FedVocModel


class Server:
    """
    FedAvg aggregation server.

    BUG FIX kept: weighted aggregation by dataset size.
    Original divided by len(clients) regardless of data size — biased the model.
    """

    def __init__(self, vocab_size):
        self.global_model = FedVocModel(vocab_size)

    def aggregate(self, client_weights_list, dataset_weights=None):
        """
        Args:
            client_weights_list: list of state_dicts from each client
            dataset_weights:     list of floats summing to 1.0
                                 If None, falls back to uniform average.
        """
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
