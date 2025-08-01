#!/bin/bash

service ssh restart

GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_np || exit 1

echo "127.0.0.1" > ip_list.txt

cat ip_list.txt

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}

echo "Test GraphStorm node classification"

date

echo "**************standalone"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --cf ml_nc.yaml

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --decoder-norm layer

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT & construct, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --node-feat-name movie:title --construct-feat-ntype user --decoder-norm batch

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT & construct, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --node-feat-name movie:title --construct-feat-ntype user --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, remove activation and dropout in the input layer"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --input-activate none

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, add bias and dropout"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --dropout 0.5 --decoder-bias True

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, no test_set"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_notest_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --logging-file /tmp/train_log.txt

error_and_exit $?

bst_cnt=$(grep "Best Test accuracy: N/A" /tmp/train_log.txt | wc -l)
if test $bst_cnt -lt 1
then
    echo "Test set is empty we should have Best Test accuracy: N/A"
    exit -1
fi

rm /tmp/train_log.txt

mkdir -p /tmp/ML_np_profile

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, with profiling"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --profile-path /tmp/ML_np_profile

error_and_exit $?

cnt=$(ls /tmp/ML_np_profile/*.csv | wc -l)
if test $cnt -lt 1
then
    echo "Cannot find the profiling files."
    exit -1
fi

rm -R /tmp/ML_np_profile

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, mlp layer between GNN layer: 1"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-mini-batch-infer false --num-ffn-layers-in-gnn 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model

# Ensure a file named GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml was created under --save-model-path
if [ ! -f ./models/movielen_100k/train_val/movielen_100k_ngnn_model/GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml ]; then
    echo "GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml was not created"
    exit 1
fi

# Ensure a file named data_transform_new.json was copied under --save-model-path
if [ ! -f ./models/movielen_100k/train_val/movielen_100k_ngnn_model/data_transform_new.json ]; then
    echo "data_transform_new.json was not copied from input data"
    exit 1
fi


error_and_exit $?

echo "**************dataset: MovieLens, Check test-set-only inference"
# Create a temp dir for embedding output
EMBED_CHECK_DIR=$(mktemp -d)
# Set up tmpdir cleanup trap
trap 'rm -rf "$EMBED_CHECK_DIR"; echo "Cleaned up temp directories"' EXIT

# Only infer test set
python3 -m graphstorm.run.gs_node_classification \
    --inference \
    --workspace $GS_HOME/training_scripts/gsgnn_np \
    --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 \
    --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json \
    --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml \
    --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-1/ \
    --save-embed-path "$EMBED_CHECK_DIR/only-test-embeddings/"

error_and_exit $?

# Ensure only test nodes have embeddings
# test set is 10% of 1682 nodes, so we expect 168 nodes
python3 $GS_HOME/tests/end2end-tests/graphstorm-nc/check_emb.py \
    --emb-path "$EMBED_CHECK_DIR/only-test-embeddings/" \
    --graph-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json \
    --ntypes "movie" \
    --expected-row-count 168 \
    --emb-size 128 \
    --file-format "parquet"

error_and_exit $?

echo "**************dataset: MovieLens, Check all-nodes inference"
# Infer all nodes
python3 -m graphstorm.run.gs_node_classification \
    --inference \
    --infer-all-target-nodes true \
    --no-validation true \
    --workspace $GS_HOME/training_scripts/gsgnn_np \
    --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 \
    --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json \
    --ip-config ip_list.txt --ssh-port 2222 \
    --cf ml_nc.yaml \
    --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-1/ \
    --save-embed-path "$EMBED_CHECK_DIR/all-node-embeddings/"

error_and_exit $?

# Ensure all nodes have embeddings
python3 $GS_HOME/tests/end2end-tests/graphstorm-nc/check_emb.py \
    --emb-path "$EMBED_CHECK_DIR/all-node-embeddings/" \
    --graph-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json \
    --ntypes "movie" \
    --emb-size 128 \
    --file-format "parquet"

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, mlp layer in input layer: 1"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-mini-batch-infer false --num-ffn-layers-in-input 1 --save-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model

error_and_exit $?

python3 -m graphstorm.run.gs_node_classification --inference --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --restore-model-path ./models/movielen_100k/train_val/movielen_100k_ngnn_model/epoch-1/

error_and_exit $?

rm -R ./models/movielen_100k/train_val/movielen_100k_ngnn_model

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-node-embeddings true --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT & sparse embed, BERT nodes: movie, inference: full-graph, gradient clip: 0.1, grad norm type: 2"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --use-node-embeddings true --use-mini-batch-infer false --max-grad-norm 0.1 --grad-norm-type 2

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false   --save-model-path ./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model --save-model-frequency 1000

error_and_exit $?

echo "**************restart training from iteration 1 of the previous training"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --fanout '10,15' --num-layers 2 --restore-model-path ./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-1

error_and_exit $?

## load emb from previous run and check its shape
python3 $GS_HOME/tests/end2end-tests/graphstorm-nc/check_emb.py --emb-path "./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-1/" --graph-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ntypes "movie user" --emb-size 128

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/graphstorm-nc/check_emb.py --emb-path "./models/movielen_100k/train_val/movielen_100k_utext_train_val_1p_4t_model/epoch-2/" --graph-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ntypes "movie user" --emb-size 128

error_and_exit $?


echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, num-heads: 8"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat --num-heads 8

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat --use-mini-batch-infer false

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, norm: batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat --use-mini-batch-infer false --gnn-norm batch

error_and_exit $?

echo "**************dataset: MovieLens, RGAT layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type rgat --fanout '5,10' --num-layers 2 --use-mini-batch-infer false

error_and_exit $?


echo "**************dataset: MovieLens, HGT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type hgt

error_and_exit $?

echo "**************dataset: MovieLens, HGT layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --model-encoder-type hgt --use-mini-batch-infer false

error_and_exit $?

rm -Rf /data/movielen_100k_multi_label_nc
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_multi_label_nc
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_label_nc --node_class 1 --field label

echo "**************dataset: multilabel MovieLens, RGCN layer: 1, node feat: generated feature, inference: mini-batch, save emb"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --save-embed-path ./model/ml-emb/ --num-epochs 3 --multilabel true --num-classes 5 --node-feat-name movie:title user:feat

error_and_exit $?

echo "**************dataset: multilabel MovieLens with weight, RGCN layer: 1, node feat: generated feature, inference: full graph, save emb"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_label_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --save-embed-path ./model/ml-emb/ --num-epochs 3 --multilabel true --num-classes 5 --node-feat-name movie:title user:feat --use-mini-batch-infer false --multilabel-weights 0.3,0.3,0.1,0.1,0.2

error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 2, node feat: fixed HF BERT, BERT nodes: movie, inference: full-graph, imbalance-class"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --imbalance-class-weights 1,1,1,1,2,1,1,1,1,2,1,1,1,1,2,1,1,1,1

error_and_exit $?

rm -Rf /data/movielen_100k_multi_node_feat_nc
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_multi_node_feat_nc
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multi_feat_nc.py --path /data/movielen_100k_multi_node_feat_nc

echo "**************dataset: multi-feature MovieLens, RGCN layer: 1, node feat: generated feature, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_node_feat_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --num-epochs 3 --node-feat-name user:feat0 movie:title

error_and_exit $?

rm -Rf /data/movielen_100k_multi_feat_nc
cp -R /data/movielen_100k_train_val_1p_4t /data/movielen_100k_multi_feat_nc
# generate a dataset with user and movie have multiple node features
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multi_feat_nc.py --path /data/movielen_100k_multi_feat_nc --multi_feats=True

echo "**************dataset: multi-feature MovieLens, RGCN layer: 1, node feat: generated feature, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_feat_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --num-epochs 3  --node-feat-name movie:title user:feat0,feat1

error_and_exit $?

echo "**************dataset: multi target ntypes MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_target_ntypes_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_multi_target_ntypes.yaml --num-epochs 3 --save-model-path /data/gsgnn_nc_ml/

error_and_exit $?

echo "**************dataset: multi target ntypes MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, do inference"
python3 -m graphstorm.run.gs_node_classification --inference --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_target_ntypes_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_multi_target_ntypes.yaml --save-embed-path /data/gsgnn_nc_ml/infer-emb/ --save-prediction-path /data/gsgnn_nc_ml/prediction/ --restore-model-path /data/gsgnn_nc_ml/epoch-2/ --preserve-input True

error_and_exit $?

cnt=$(ls -l /data/gsgnn_nc_ml/infer-emb/ | wc -l)
if test $cnt != 4
then
    echo "We save both movie and user"
    exit -1
fi

cnt=$(ls -l /data/gsgnn_nc_ml/infer-emb/movie/ | grep "embed-" | wc -l)
cnt=$(($cnt/2))
if test $cnt != $NUM_TRAINERS
then
    echo "There must be $NUM_TRAINERS embedding parts."
    exit -1
fi

cnt=$(ls -l /data/gsgnn_nc_ml/infer-emb/user/ | grep "embed-" | wc -l)
cnt=$(($cnt/2))
if test $cnt != $NUM_TRAINERS
then
    echo "There must be $NUM_TRAINERS embedding parts."
    exit -1
fi

cnt=$(ls -l /data/gsgnn_nc_ml/prediction/| wc -l)
if test $cnt != 4
then
    echo "We save prediction results of movie and user."
    exit -1
fi

cnt=$(ls -l /data/gsgnn_nc_ml/prediction/movie/ | grep "predict" | wc -l)
if test $cnt != $NUM_TRAINERS * 2
then
    echo "There must be $NUM_TRAINERS * 2 prediction parts for movie as --preserve-input is True."
    exit -1
fi

cnt=$(ls -l /data/gsgnn_nc_ml/prediction/movie/ | grep "nids" | wc -l)
if test $cnt != $NUM_TRAINERS
then
    echo "There must be $NUM_TRAINERS prediction parts for movie."
    exit -1
fi

cnt=$(ls -l /data/gsgnn_nc_ml/prediction/user/ | grep "predict" | wc -l)
if test $cnt != $NUM_TRAINERS * 2
then
    echo "There must be $NUM_TRAINERS * 2 prediction parts for user as --preserve-input is True."
    exit -1
fi

cnt=$(ls -l /data/gsgnn_nc_ml/prediction/user/ | grep "nids" | wc -l)
if test $cnt != $NUM_TRAINERS
then
    echo "There must be $NUM_TRAINERS prediction parts for user."
    exit -1
fi

rm -fr /data/gsgnn_nc_ml/*

echo "**************dataset: multi target ntypes multiclass MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch with tensorboard tracker"
# generate a dataset with user and movie have multilabel
python3 $GS_HOME/tests/end2end-tests/data_gen/gen_multilabel.py --path /data/movielen_100k_multi_target_ntypes_train_val_1p_4t --node_class 1 --field label
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_target_ntypes_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_multi_target_ntypes_multilabel.yaml --num-epochs 3
error_and_exit $?

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, focal loss with tensorboard tracker but not tensorboard package installed"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --decoder-norm layer --class-loss-func focal --num-classes 2 --task-tracker tensorboard_task_tracker

if test $? -eq 0
then
    # Tensorboard package is not installed.
    # The above command should fail.
    exit -1
fi

# Install tensorboard package
python3 -m pip install tensorboard

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, focal loss with tensorboard tracker"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --decoder-norm layer --class-loss-func focal --num-classes 2 --task-tracker tensorboard_task_tracker

error_and_exit $?

cnt=$(ls -l $GS_HOME/training_scripts/gsgnn_np/runs/| wc -l)
if test $cnt -ne 2
then
    echo "Tensorboard logs must be stored under ./runs/"
    exit -1
fi
rm -fr $GS_HOME/training_scripts/gsgnn_np/runs/

echo "**************dataset: MovieLens, RGCN layer: 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, focal loss with tensorboard tracker"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --decoder-norm layer --class-loss-func focal --num-classes 2 --task-tracker tensorboard_task_tracker:./logs/

error_and_exit $?

cnt=$(ls -l $GS_HOME/training_scripts/gsgnn_np/logs/| wc -l)
if test $cnt -ne 2
then
    echo "Tensorboard logs must be stored under ./logs/"
    exit -1
fi
rm -fr $GS_HOME/training_scripts/gsgnn_np/logs/

echo "**************dataset: multi-feature MovieLens, RGCN layer: 1, node feat: generated feature, inference: mini-batch, multiple feat groups"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_node_feat_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --num-epochs 3 --node-feat-name user:feat0 movie:title user:feat0

error_and_exit $?

echo "**************dataset: multi-feature MovieLens, RGCN layer: 1, node feat: generated feature, inference: full graph, multiple feat groups with learnable embeddings "
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_multi_node_feat_nc/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --num-epochs 3 --node-feat-name user:feat0 movie:title user:feat0 --use-node-embeddings true --use-mini-batch-infer false

error_and_exit $?

date

echo 'Done'
