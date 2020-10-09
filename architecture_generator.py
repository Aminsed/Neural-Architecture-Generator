import random
import networkx as nx
import tensorflow as tf
from graph_utils import draw_graph
from model_utils import convert_to_tensorflow

class RandomArchitectureGenerator:
    def __init__(self, input_shape, num_classes, max_layers, max_neurons, skip_connection_prob=0.3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_layers = max_layers
        self.max_neurons = max_neurons
        self.skip_connection_prob = skip_connection_prob

    def create_random_architecture(self):
        G = nx.DiGraph()
        self._add_input_layer(G)
        num_layers = self._generate_random_num_layers()
        self._add_hidden_layers(G, num_layers)
        self._add_output_layer(G, num_layers)
        self._ensure_outgoing_connections(G, num_layers)
        self._connect_output_neurons(G, num_layers)
        self._draw_graph(G)
        return G

    def _add_input_layer(self, G):
        num_input_neurons = self.input_shape[0]
        for i in range(num_input_neurons):
            input_neuron = f"Input_{i+1}"
            G.add_node(input_neuron, layer=0)

    def _generate_random_num_layers(self):
        return random.randint(1, self.max_layers)

    def _add_hidden_layers(self, G, num_layers):
        for i in range(num_layers):
            num_neurons = random.randint(1, self.max_neurons)
            for j in range(num_neurons):
                neuron_name = f"Neuron_{i+1}_{j+1}"
                G.add_node(neuron_name, layer=i+1)
                self._connect_neuron(G, i, neuron_name)
                self._add_skip_connections(G, i, neuron_name)

    def _connect_neuron(self, G, layer_idx, neuron_name):
        if layer_idx == 0:
            prev_layer_neurons = [node for node in G.nodes if 'Input' in node]
        else:
            prev_layer_neurons = [node for node in G.nodes if G.nodes[node].get('layer') == layer_idx]
        num_connections = random.randint(1, len(prev_layer_neurons))
        for _ in range(num_connections):
            prev_neuron = random.choice(prev_layer_neurons)
            G.add_edge(prev_neuron, neuron_name)

    def _add_skip_connections(self, G, layer_idx, neuron_name):
        if layer_idx > 0 and random.random() < self.skip_connection_prob:
            skip_layer = random.randint(0, layer_idx-1)
            skip_layer_neurons = [node for node in G.nodes if G.nodes[node].get('layer') == skip_layer]
            if skip_layer_neurons:
                skip_neuron = random.choice(skip_layer_neurons)
                G.add_edge(skip_neuron, neuron_name)

    def _add_output_layer(self, G, num_layers):
        for i in range(self.num_classes):
            output_neuron = f"Output_{i+1}"
            G.add_node(output_neuron, layer=num_layers+1)

    def _ensure_outgoing_connections(self, G, num_layers):
        for i in range(num_layers):
            curr_layer_neurons = [node for node in G.nodes if G.nodes[node].get('layer') == i+1]
            next_layer_neurons = [node for node in G.nodes if G.nodes[node].get('layer') == i+2] if i < num_layers - 1 else [node for node in G.nodes if 'Output' in node]
            for neuron in curr_layer_neurons:
                if not G.out_edges(neuron):
                    next_neuron = random.choice(next_layer_neurons)
                    G.add_edge(neuron, next_neuron)

    def _connect_output_neurons(self, G, num_layers):
        prev_layer_neurons = [node for node in G.nodes if G.nodes[node].get('layer') == num_layers]
        for output_neuron in [node for node in G.nodes if 'Output' in node]:
            num_connections = random.randint(1, len(prev_layer_neurons))
            for _ in range(num_connections):
                prev_neuron = random.choice(prev_layer_neurons)
                G.add_edge(prev_neuron, output_neuron)

    def _draw_graph(self, G):
        draw_graph(G)

    def convert_to_tensorflow(self, G):
        return convert_to_tensorflow(G, self.input_shape)