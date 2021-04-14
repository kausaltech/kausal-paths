import dvc_pandas
from .datasets import Dataset


class Context:
    nodes: dict
    datasets: dict[str, Dataset]

    def __init__(self):
        self.nodes = {}
        self.datasets = {}

    def load_dataset(self, identifier):
        if identifier in self.datasets:
            return self.datasets[identifier]
        df = dvc_pandas.load_dataset(identifier)
        self.datasets[identifier] = df
        return df

    def add_dataset(self, config):
        assert config['identifier'] not in self.datasets
        ds = Dataset(**config)
        df = ds.load(self)
        self.datasets[config['identifier']] = df
