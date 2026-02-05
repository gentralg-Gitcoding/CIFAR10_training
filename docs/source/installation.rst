Installation
============

The recommended way to use CIFAR-10 Tools is through the provided development container, 
which includes all dependencies and development tools pre-configured.

Using the development container
--------------------------------

The repository includes a complete development container configuration that provides:

* Python 3.10+ environment
* PyTorch with CUDA support
* All required dependencies
* Jupyter notebook support

**Prerequisites:**

* `Docker <https://www.docker.com/get-started>`_
* `Visual Studio Code <https://code.visualstudio.com/>`_
* `Dev Containers extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_

**Steps:**

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/gperdrizet/CIFAR10.git
      cd CIFAR10

2. Open in VS Code:

   .. code-block:: bash

      code .

3. When prompted, click **"Reopen in Container"** or use the command palette 
   (``Ctrl+Shift+P`` / ``Cmd+Shift+P``) and select **"Dev Containers: Reopen in Container"**

4. Wait for the container to build (first time only)

5. You're ready to go! All dependencies are installed and configured.

Installing the package only
----------------------------

If you only need the ``cifar10_tools`` package without the demo notebooks and full development environment, install from PyPI:

.. code-block:: bash

   pip install cifar10_tools

Dependencies
------------

Core dependencies for ``cifar10_tools`` package (automatically installed):

* ``python`` >= 3.10, < 3.13
* ``torch`` >= 2.0
* ``torchvision`` >= 0.15
* ``numpy`` >= 1.24

Optional dependencies for notebooks (preinstalled in devcontainer):

* ``optuna`` - Hyperparameter optimization
* ``matplotlib`` - Plotting and visualization
* ``scikit-learn`` - Evaluation metrics


Verifying Installation
----------------------

To verify your installation:

.. code-block:: python

   import cifar10_tools
   from cifar10_tools.pytorch.data import make_data_loaders
   
   print(f'cifar10_tools version: {cifar10_tools.__version__}')

GPU Support
-----------

The development container is configured for GPU support. To verify CUDA availability:

.. code-block:: python

   import torch
   
   print(f'CUDA available: {torch.cuda.is_available()}')
   print(f'CUDA version: {torch.version.cuda}')
   print(f'Device count: {torch.cuda.device_count()}')

Next Steps
----------

* :doc:`quickstart` - Get started with a quick example
* :doc:`notebooks/index` - Explore the example notebooks
* :doc:`api/index` - Browse the API reference
