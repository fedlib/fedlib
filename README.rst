
.. .. raw:: html

..     <div style="text-align: center;">

.. container:: badges

    .. image:: https://img.shields.io/badge/arXiv-2206.05359-red?logo=arxiv&style=flat-square&link=https%3A%2F%2Farxiv.org%2Fpdf%2F2206.05359.pdf
        :alt: Static Badge
        :target: https://arxiv.org/pdf/2206.05359.pdf

    .. image:: https://img.shields.io/github/last-commit/lishenghui/blades/master?logo=Github
        :alt: GitHub last commit (branch)
        :target: https://github.com/lishenghui/blades

    .. image:: https://img.shields.io/github/actions/workflow/status/lishenghui/blades/.github%2Fworkflows%2Funit-tests.yml?logo=Pytest&logoColor=hsl&label=Unit%20Testing
       :alt: GitHub Workflow Status (with event)

    .. image:: https://img.shields.io/badge/Pytorch-2.0-brightgreen?logo=pytorch&logoColor=red
       :alt: Static Badge
       :target: https://pytorch.org/get-started/pytorch-2.0/

    .. image:: https://img.shields.io/badge/Ray-2.8-brightgreen?logo=ray&logoColor=blue
       :alt: Static Badge
       :target: https://docs.ray.io/en/releases-2.8.0/

    .. image:: https://readthedocs.org/projects/blades/badge/?version=latest
        :target: https://blades.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

    .. image:: https://img.shields.io/github/license/lishenghui/blades?logo=apache&logoColor=red
        :alt: GitHub
        :target: https://github.com/lishenghui/blades/blob/master/LICENSE


.. .. raw:: html

..     <p align=center>
..         <img src="https://github.com/lishenghui/blades/raw/master/docs/source/images/arch.png" width="1000" alt="Blades Logo">
..     </p>

.. image:: https://github.com/lishenghui/blades/raw/master/docs/source/images/arch.png



Installation
==================================================

.. code-block:: bash

    git clone https://github.com/lishenghui/blades
    cd blades
    pip install -v -e .
    # "-v" means verbose, or more output
    # "-e" means installing a project in editable mode,
    # thus any local modifications made to the code will take effect without reinstallation.


.. code-block:: bash

    cd blades/blades
    python train.py file ./tuned_examples/fedsgd_cnn_fashion_mnist.yaml


**Blades** internally calls `ray.tune <https://docs.ray.io/en/latest/tune/tutorials/tune-output.html>`_; therefore, the experimental results are output to its default directory: ``~/ray_results``.

Experiment Results
==================================================

.. image:: https://github.com/lishenghui/blades/raw/master/docs/source/images/fashion_mnist.png

.. image:: https://github.com/lishenghui/blades/raw/master/docs/source/images/cifar10.png




Cluster Deployment
===================

To run **blades** on a cluster, you only need to deploy ``Ray cluster`` according to the `official guide <https://docs.ray.io/en/latest/cluster/user-guide.html>`_.


Built-in Implementations
==================================================
In detail, the following strategies are currently implemented:



Data Partitioners:
==================================================

Dirichlet Partitioner
----------------------

.. image:: https://github.com/lishenghui/blades/raw/master/docs/source/images/dirichlet_partition.png

Sharding Partitioner
----------------------

.. image:: https://github.com/lishenghui/blades/raw/master/docs/source/images/shard_partition.png


Citation
=========

Please cite our `paper <https://arxiv.org/abs/2206.05359>`_ (and the respective papers of the methods used) if you use this code in your own work:

::

   @article{li2023blades,
     title={Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning},
     author= {Li, Shenghui and Ju, Li and Zhang, Tianru and Ngai, Edith and Voigt, Thiemo},
     journal={arXiv preprint arXiv:2206.05359},
     year={2023}
   }
