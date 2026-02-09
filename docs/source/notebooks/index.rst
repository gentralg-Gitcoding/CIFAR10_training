Example notebooks
=================

This section contains a progressive series of Jupyter notebooks demonstrating the capabilities 
of the CIFAR-10 Tools package. Each notebook builds on the previous one, showcasing different 
techniques and features.

The notebooks serve as practical examples of using the ``image_classification_tools`` package for real-world 
image classification tasks. They demonstrate data loading, model architecture design, training, 
hyperparameter optimization, and evaluation.

Notebook series
---------------

The following notebooks are available in the `notebooks/ directory <https://github.com/gperdrizet/CIFAR10/tree/main/notebooks>`_ of the repository:

1. `01-DNN.ipynb <https://github.com/gperdrizet/CIFAR10/blob/main/notebooks/01-DNN.ipynb>`_ - **DNN Baseline**
   
   Start with a simple fully-connected Deep Neural Network on grayscale CIFAR-10 images. 
   This establishes a baseline and demonstrates the basic data loading and training workflow.
   
   * **Key concepts**: Data loading, basic training loop, evaluation
   * **Typical accuracy**: ~45-50%

.. raw:: html

   <div style="margin-bottom: 1.5em;"></div>

2. `02-CNN.ipynb <https://github.com/gperdrizet/CIFAR10/blob/main/notebooks/02-CNN.ipynb>`_ - **Basic CNN**
   
   Introduce convolutional layers and pooling to improve upon the DNN baseline. 
   Still uses grayscale images to focus on architectural improvements.
   
   * **Key concepts**: Convolutional layers, pooling, feature extraction
   * **Typical accuracy**: ~55-60%

.. raw:: html

   <div style="margin-bottom: 1.5em;"></div>

3. `03-RGB-CNN.ipynb <https://github.com/gperdrizet/CIFAR10/blob/main/notebooks/03-RGB-CNN.ipynb>`_ - **RGB CNN**
   
   Leverage color information by using all three RGB channels. Demonstrates the importance 
   of using appropriate data preprocessing for the task.
   
   * **Key concepts**: Multi-channel inputs, color-aware features
   * **Typical accuracy**: ~65-70%

.. raw:: html

   <div style="margin-bottom: 1.5em;"></div>

4. `04-optimized-CNN.ipynb <https://github.com/gperdrizet/CIFAR10/blob/main/notebooks/04-optimized-CNN.ipynb>`_ - **Optimized CNN**
   
   Use Optuna for automated hyperparameter search to find optimal model configuration. 
   Demonstrates the power of systematic hyperparameter tuning.
   
   * **Key concepts**: Hyperparameter optimization, search spaces, Optuna integration
   * **Typical accuracy**: ~75-80%

.. raw:: html

   <div style="margin-bottom: 1.5em;"></div>

5. `05-augmented-CNN.ipynb <https://github.com/gperdrizet/CIFAR10/blob/main/notebooks/05-augmented-CNN.ipynb>`_ - **Augmented CNN**
   
   Apply data augmentation techniques to the optimized architecture for improved 
   generalization and final performance.
   
   * **Key concepts**: Data augmentation, regularization, generalization
   * **Typical accuracy**: ~80-85%


Running the Notebooks
---------------------

All notebooks are designed to run in the provided development container:

1. Open the repository in VS Code with the Dev Containers extension
2. Reopen in container when prompted
3. Navigate to the ``notebooks/`` directory
4. Open any notebook and run cells sequentially

Each notebook is self-contained and includes:

* Detailed explanations of concepts
* Complete working code
* Visualization of results
* Performance analysis

Common Patterns
---------------

All notebooks follow a similar structure:

1. **Setup**: Import packages, configure device, set random seeds
2. **Data Loading**: Load and preprocess CIFAR-10 data
3. **Model Definition**: Create neural network architecture
4. **Training**: Train model with appropriate hyperparameters
5. **Evaluation**: Assess performance on test set
6. **Visualization**: Plot results and analyze model behavior

This consistent structure makes it easy to compare approaches and understand 
the impact of each technique.
