import importlib
import yaml
from nodes.base import Dataset


class InstanceLoader:
    def make_node(self, node_class, config):
        ds_config = config.get('input_datasets', [])
        datasets = []
        for ds in ds_config:
            o = Dataset(identifier=ds['identifier'], column=ds.get('column'))
            datasets.append(o)
        node = node_class(config['identifier'], input_datasets=datasets)
        node.config = config
        return node

    def setup_nodes(self):
        self.nodes = {}

        for nc in self.config['nodes']:
            klass = nc['type'].split('.')
            node_name = klass.pop(-1)
            klass.insert(0, 'nodes')
            mod = importlib.import_module('.'.join(klass))
            node_class = getattr(mod, node_name)
            node = self.make_node(node_class, nc)
            assert node.identifier not in self.nodes
            self.nodes[node.identifier] = node

        # Setup edges
        for node in self.nodes.values():
            for out_id in node.config.get('output_nodes', []):
                out_node = self.nodes[out_id]
                assert out_id in self.nodes
                out_node.input_nodes.append(node)
                node.output_nodes.append(out_node)
                # FIXME: Check for cycles?

    def print_graph(self, node=None, indent=0):
        if node is None:
            all_nodes = self.nodes.values()
            root_nodes = list(filter(lambda node: not node.output_nodes, all_nodes))
            assert len(root_nodes) == 1
            node = root_nodes[0]

        print('  ' * indent + str(node))
        for in_node in node.input_nodes:
            self.print_graph(in_node, indent + 1)

    def __init__(self, fn):
        data = yaml.load(open('tampere.yaml', 'r'), Loader=yaml.Loader)
        self.config = data['instance']
        self.setup_nodes()


loader = InstanceLoader('tampere.yaml')
loader.print_graph()
