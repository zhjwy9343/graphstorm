.. _gs-output-remapping:

GraphStorm Output Node ID Remapping
====================================

As explained in the :ref:`outputs of graph construction <gcon-output-format>` guide, The original node IDs, which are normally strings, provided by users in the raw input data tables will be converted into integer-based IDs first, and then be shuffled during graph partition operations, which will create another set of integer node IDs used in model training and inference. When saving outputs, e.g., node embeddings or predictions, of training and inference, these outputs are based on the shuffled node IDs.

In general, these outputs are not directly consumable due to the node ID difference between the original node IDs user provided and the shuffled node IDs used in model training and inference. It is required to conduct node ID remapping to make sure output results map to the correct raw node IDs. This document provides the guidline of how to do the remapping.

.. note::
    
    * The built-in GraphStorm training or inference pipeline (launched by GraphStorm CLIs) will automatically do the node ID remapping, aligning the embeddings or predictions to their corresponding raw node IDs, and saving them as a dataframe to parquet files.

Remapping inputs
-----------------
Inputs of the node ID remapping are the :ref:`outputs of GraphStorm model training and inference <gs-output>`, which normally include two sets of ``.pt`` files. One set includes files of shuffled integer node IDs, while another includes the corresponding embeddings or predictions in the same row order as the node ID files. 

Below box shows a simple example of node prediction output files.

.. code-block:: bash

    # Saved embedding files and node ID files
    save_embed_path_dir/
        ntype0/
            embed_nids-00000.pt
            embed_nids-00001.pt
            embed-00000.pt
            embed-00001.pt
        ntype1/
            embed_nids-00000.pt
            embed_nids-00001.pt
            embed-00000.pt
            embed-00001.pt

    # Saved prediction files and node ID files
    save_prediction_path_dir/
        ntype0/
            predict_nids-00000.pt
            predict_nids-00001.pt
            predict-00000.pt
            predict-00001.pt
        ntype1/
            predict_nids-00000.pt
            predict_nids-00001.pt
            predict-00000.pt
            predict-00001.pt
    
    # embedding or prediction file contents and node ID fiel contents

    embed_nids-00000.pt | embed-00000.pt                predict_nids-00000.pt | predict.pt
                        |                                                     |
    Graph Node ID       |   embeddings                  Graph Node ID         |   Prediction results
    10                  |   0.112,0.123,-0.011,...      10                    |   0
    1                   |   0.872,0.321,-0.901,...      1                     |   0
    23                  |   0.472,0.432,-0.732,...      23                    |   1

Another set of important input data for remapping is the node and edge mapping files generated and saved by graph construction guide.

Remapping CLIs
---------------

Remapping outputs
------------------

