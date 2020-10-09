# Random Neural Network Architecture Generator

This project generates random neural network architectures using Python and the NetworkX library. The generated architectures can be 
visualized and converted to TensorFlow models for further use.

## Project Structure

The project is organized into the following files:

- `demo.ipynb`: The Jupyter notebook that demonstrates how to use the `RandomArchitectureGenerator` class.
- `architecture_generator.py`: Contains the `RandomArchitectureGenerator` class and its associated methods for generating random 
neural network architectures.
- `graph_utils.py`: Contains utility functions for visualizing the generated architectures using NetworkX and Matplotlib.
- `model_utils.py`: Contains utility functions for converting the generated architectures to TensorFlow models.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- NetworkX
- Matplotlib
- TensorFlow

You can install the required packages using pip:
```
pip install networkx matplotlib tensorflow
```

## Usage

To use the Random Neural Network Architecture Generator, follow these steps:

1. Clone the repository

2. Run the `demo.ipynb` notebook cells:


This will generate a random neural network architecture, visualize it using NetworkX and Matplotlib, and convert it to a TensorFlow 
model. The model summary will be printed to the console.

You can customize the input shape, number of classes, maximum number of layers, maximum number of neurons, and skip connection 
probability by modifying the corresponding variables in the `demo.ipynb` notebook.

## Customization

The `RandomArchitectureGenerator` class provides several parameters that you can customize:

- `input_shape`: The shape of the input data.
- `num_classes`: The number of output classes.
- `max_layers`: The maximum number of hidden layers in the generated architecture.
- `max_neurons`: The maximum number of neurons per hidden layer.
- `skip_connection_prob`: The probability of adding skip connections between layers.

You can modify these parameters when creating an instance of the `RandomArchitectureGenerator` class in the `demo.ipynb` script.



## Acknowledgements

This project was inspired by the need for exploring random neural network architectures and the desire to automate the process of 
generating and visualizing them.

## Contact

For any questions or inquiries, please contact [amin.sedaghat.ext@gmail.com](mailto:amin.sedaghat.ext@gmail.com).
