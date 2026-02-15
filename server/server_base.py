import copy
from models.base_model import DistilBertLM


class Server:
    def __init__(self, vocab_size):
        self.global_model = DistilBertLM(vocab_size)

    def aggregate(self, client_weights):
        new_state = copy.deepcopy(self.global_model.state_dict())

        for key in new_state.keys():
            new_state[key] = sum(
                weights[key] for weights in client_weights
            ) / len(client_weights)

        self.global_model.load_state_dict(new_state)
