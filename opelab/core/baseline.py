from opelab.core.data import DataType
from opelab.core.policy import Policy


class Baseline:
    
    def evaluate(self, data:DataType, target:Policy, behavior:Policy, gamma:float=1.0, reward_estimator=None) -> float:
        raise NotImplementedError
    
    def load_data(self, data):
        pass
