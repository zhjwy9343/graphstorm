.. _gs-output-remapping:

GraphStorm Output Node ID Remapping
====================================

As explained in the :ref:`outputs of graph construction <gs-id-mapping-files>` guide, the original node IDs provided by users in the raw input data tables, which are normally strings, will be converted into integer-based IDs first, and then be shuffled during graph partition operations, which will create another set of integer node IDs used in model training and inference. When saving outputs of training and inference, e.g., node embeddings or predictions, these outputs are ordered based on the shuffled node IDs.

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

Another set of important input data for remapping is the :ref:`node and edge mapping files <gs-id-mapping-files>`  generated and saved by graph construction pipelines.

Remapping CLI
---------------

A simple GraphStorm remapping CLI template is like the command below.

.. code:: bash

    python -m graphstorm.gconstruct.remap_result \
              --node-id-mapping path_to_id_mapping_folder/ \
              --node-emb-dir path_to_save_embed_path_dir/  \
              --prediction-dir path_to_save_prediction_path_dir/\
              --preserve-input True \
              --output-format csv

In this CLI, the **-\-node-id-mapping** argument specifies the folder that stores the node and edge mapping files, and the **-\-node-emb-dir** and **-\-prediction-dir** arguments indicate the folder of saved embeddings, and saved prediction results, respectively.

By default, the remapping CLI will remove the saved files of embeddings or predictions after remapping operation. If users want to keep these embeddings or predictions, you can set the argument **--preserve-input** to be ``True`` to fulfill this requirement.

Another important argument of this remapping CLI is the **-\-output-format**, which specifies the remapping output file format. By default, outputs are saved in ``parquet`` format. Set to ``csv`` can save outputs in CSV format.

Remapping outputs
------------------


Remapping CLI configurations
-----------------------------
