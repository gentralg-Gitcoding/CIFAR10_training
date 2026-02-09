Plotting
========

.. automodule:: image_classification_tools.pytorch.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The plotting module provides comprehensive visualization utilities:

* **Training curves**: Loss and accuracy over epochs
* **Confusion matrices**: Classification performance heatmaps
* **Sample images**: Visualize dataset examples
* **Probability distributions**: Class prediction confidence
* **Evaluation curves**: ROC and precision-recall curves
* **Optimization results**: Optuna trial visualizations

All functions return matplotlib figure and axes objects for further customization.

Example usage
-------------

Learning curves:

.. code-block:: python

   from image_classification_tools.pytorch.plotting import plot_learning_curves
   import matplotlib.pyplot as plt

   # After training
   fig, axes = plot_learning_curves(history)
   plt.show()

Confusion matrix:

.. code-block:: python

   from image_classification_tools.pytorch.plotting import plot_confusion_matrix

   fig, ax = plot_confusion_matrix(true_labels, predictions, class_names)
   plt.title('Test Set Confusion Matrix')
   plt.show()

Sample images:

.. code-block:: python

   from image_classification_tools.pytorch.plotting import plot_sample_images

   fig, axes = plot_sample_images(dataset, class_names, nrows=2, ncols=5)
   plt.show()

Class probability distributions:

.. code-block:: python

   from image_classification_tools.pytorch.plotting import plot_class_probability_distributions

   # Get predicted probabilities
   model.eval()
   all_probs = []
   with torch.no_grad():
       for images, _ in test_loader:
           outputs = model(images)
           probs = torch.softmax(outputs, dim=1)
           all_probs.append(probs.cpu().numpy())
   
   all_probs = np.concatenate(all_probs, axis=0)
   
   fig, axes = plot_class_probability_distributions(all_probs, class_names)
   plt.show()

Evaluation curves:

.. code-block:: python

   from image_classification_tools.pytorch.plotting import plot_evaluation_curves

   fig, (ax1, ax2) = plot_evaluation_curves(true_labels, all_probs, class_names)
   plt.show()

Optimization results:

.. code-block:: python

   from image_classification_tools.pytorch.plotting import plot_optimization_results

   # After Optuna study
   fig, axes = plot_optimization_results(study)
   plt.show()
