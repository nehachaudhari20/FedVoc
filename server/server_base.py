import copy
from models.base_model import FedVocModel


class Server:
    """
    FedAvg aggregation server.

    Changes from original:
    - aggregate() now accepts optional per-client weights for weighted FedAvg
      (weighted by dataset size — standard FedAvg formula)
    - Defaults to uniform weighting if no weights provided (backwards compatible)
    """

    def __init__(self, vocab_size):
        self.global_model = FedVocModel(vocab_size)

    def aggregate(self, client_weights_list, dataset_weights=None):
        """
        Aggregate client model weights into the global model.

        Args:
            client_weights_list: list of state_dicts from each client
            dataset_weights: list of floats summing to 1.0 (dataset-size weights).
                             If None, uniform averaging is used.
        """
        n = len(client_weights_list)

        if dataset_weights is None:
            dataset_weights = [1.0 / n] * n

        new_state = copy.deepcopy(self.global_model.state_dict())

        for key in new_state.keys():
            # Skip vocab-specific layers — they differ per client in FedVoc
            # but in FedAvg all clients share the same vocab, so we average everything
            # except the embedding and lm_head (which use the global vocab anyway)
            new_state[key] = sum(
                w * weights[key]
                for w, weights in zip(dataset_weights, client_weights_list)
            )

        self.global_model.load_state_dict(new_state)
