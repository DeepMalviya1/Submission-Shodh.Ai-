
# Logic Gate Neural Network Report

## Model Architecture
We implemented a neural network to learn the XOR logic gate. The neural network architecture is as follows:

- **Input Layer:** 2 neurons (for the two input features of the XOR gate).
- **Hidden Layer:** 2 neurons with ReLU activation function.
- **Output Layer:** 1 neuron with Sigmoid activation function.

The ReLU activation introduces non-linearity, enabling the model to learn the XOR function, which is not linearly separable.

## Training Process
1. **Dataset Preparation:**
   - Inputs: `[[0, 0], [0, 1], [1, 0], [1, 1]]`
   - Outputs: `[[0], [1], [1], [0]]` (corresponding XOR outputs).

2. **Model Initialization:**
   - Loss Function: Binary Cross-Entropy Loss (BCELoss), suitable for binary classification.
   - Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.1.

3. **Training Loop:**
   - The model was trained for 10,000 epochs.
   - In each epoch, we performed a forward pass, computed the loss, performed a backward pass to calculate gradients, and updated the model parameters using the optimizer.

4. **Evaluation:**
   - After training, the model's predictions on the training data were: `[[0.01], [0.99], [0.99], [0.01]]`.
   - The predictions are close to the expected outputs `[0, 1, 1, 0]`, indicating the model successfully learned the XOR function.

## Results
The neural network successfully learned the XOR logic gate. The decision boundary plot illustrates how the model separates the input space into two regions, corresponding to the XOR output values.

## Instructions for Using the Visualisation Interface
1. **Launch the Interface:**
   - Run the last cell in the provided Jupyter Notebook to start the Gradio interface.
   - The interface allows users to input two binary values (0 or 1) and see the predicted XOR output.

2. **Interactive Interface:**
   - The user inputs the values in the provided fields and clicks the "Submit" button.
   - The model processes the input and displays the predicted output.

## Visualisation Link
The interactive visualisation can be demonstrated using the provided Jupyter Notebook file: [logic_gate_nn.ipynb](logic_gate_nn.ipynb).

## Contact Information
- **Name:** Deep Malviya
- **Email:** deepmalviya.aie@gmail.com
