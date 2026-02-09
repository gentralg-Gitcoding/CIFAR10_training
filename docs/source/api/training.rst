Training
========

.. automodule:: image_classification_tools.pytorch.training
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: image_classification_tools.pytorch.training.train_model

Overview
--------

The training module provides utilities for model training with:

* Progress tracking and reporting
* Training/validation loss and accuracy logging
* Configurable print frequency
* Optional device specification
* Training history returned as a dictionary

Example usage
-------------

Basic training:

.. code-block:: python

   import torch
   from image_classification_tools.pytorch.training import train_model

   model = create_model()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       epochs=50,
       print_every=5
   )

   # Access training history
   print(f"Final train accuracy: {history['train_accuracy'][-1]:.2f}%")
   print(f"Final val accuracy: {history['val_accuracy'][-1]:.2f}%")

With custom device and scheduler:

.. code-block:: python

   device = torch.device('cuda')
   model = create_model().to(device)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

   history = train_model(
       model=model,
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=criterion,
       optimizer=optimizer,
       epochs=100,
       print_every=10,
       device=device
   )
