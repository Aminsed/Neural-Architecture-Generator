import tensorflow as tf
from typing import Dict, Any

def convert_to_tensorflow(G: Dict[Any, Dict[str, Any]], input_shape: tuple) -> tf.keras.Sequential:
    """
    Converts a graph representation (G) into a TensorFlow Keras Sequential model.

    Args:
        G:  A dictionary representing the graph structure. Nodes of the graph 
            are expected to have a 'layer' attribute (if not an output node) 
            indicating their layer number.
        input_shape: A tuple defining the input shape for the model.

    Returns:
        A tf.keras.Sequential model representing the converted graph.
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    num_layers = max(G.nodes[node].get('layer', 0)  # Handle nodes without 'layer'
                     for node in G.nodes if 'Output' not in node)

    for i in range(1, num_layers + 1):
        layer_neurons = [node for node in G.nodes if G.nodes[node].get('layer') == i]
        num_neurons = len(layer_neurons)
        model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))

    num_output_neurons = len([node for node in G.nodes if 'Output' in node])
    model.add(tf.keras.layers.Dense(num_output_neurons, activation='softmax'))

    return model
