"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
import shutil
import tempfile
from unittest import mock

import pytest

from graphstorm_processing.distributed_executor import DistributedExecutor, ExecutorConfig
from graphstorm_processing.constants import TRANSFORMATIONS_FILENAME, FilesystemType, ExecutionEnv
from test_dist_heterogenous_loader import verify_integ_test_output, NODE_CLASS_GRAPHINFO_UPDATES

pytestmark = pytest.mark.usefixtures("spark")
_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(autouse=True, name="tempdir")
def tempdir_fixture():
    """Create temp dir for output files"""
    tempdirectory = tempfile.mkdtemp(
        prefix=os.path.join(_ROOT, "resources/test_output/"),
    )
    yield tempdirectory
    shutil.rmtree(tempdirectory)


def precomp_json_file(local_input, precomp_filename):
    """Copy precomputed json to local input dir"""
    precomp_file = shutil.copy(
        os.path.join(_ROOT, "resources", "precomputed_transformations", precomp_filename),
        os.path.join(local_input, TRANSFORMATIONS_FILENAME),
    )
    return precomp_file


@pytest.fixture(name="user_state_categorical_precomp_file")
def user_state_categorical_precomp_file_fixture():
    """Copy precomputed user->state feature transformation to local input dir"""
    precomp_file = precomp_json_file(
        os.path.join(_ROOT, "resources/small_heterogeneous_graph"),
        "user_state_categorical_transformation.json",
    )

    yield precomp_file

    os.remove(precomp_file)


@pytest.fixture(name="executor_configuration")
def executor_config_fixture(tempdir: str):
    """Create a re-usable ExecutorConfig"""
    input_path = os.path.join(_ROOT, "resources/small_heterogeneous_graph")
    executor_configuration = ExecutorConfig(
        local_config_path=input_path,
        local_metadata_output_path=tempdir,
        input_prefix=input_path,
        output_prefix=tempdir,
        num_output_files=-1,
        config_filename="gsprocessing-config.json",
        execution_env=ExecutionEnv.LOCAL,
        filesystem_type=FilesystemType.LOCAL,
        add_reverse_edges=True,
        graph_name="small_heterogeneous_graph",
        do_repartition=True,
    )

    yield executor_configuration


def test_dist_executor_run_with_precomputed(
    tempdir: str,
    user_state_categorical_precomp_file: str,
    executor_configuration: ExecutorConfig,
):
    """Test run function with local data"""
    original_precomp_file = user_state_categorical_precomp_file

    with open(original_precomp_file, "r", encoding="utf-8") as f:
        original_transformations = json.load(f)

    dist_executor = DistributedExecutor(executor_configuration)

    # Mock the SparkContext stop() function to leave the Spark context running
    # for the other tests, otherwise dist_executor stops it
    dist_executor.spark.stop = mock.MagicMock(name="stop")

    dist_executor.run()

    with open(os.path.join(tempdir, "metadata.json"), "r", encoding="utf-8") as mfile:
        metadata = json.load(mfile)

    verify_integ_test_output(metadata, dist_executor.loader, NODE_CLASS_GRAPHINFO_UPDATES)

    with open(os.path.join(tempdir, TRANSFORMATIONS_FILENAME), "r", encoding="utf-8") as f:
        reapplied_transformations = json.load(f)

    # There should be no difference between original and
    # pre-existing, pre-applied transformation dicts
    node_feature_transforms = original_transformations["node_features"]
    for node_type, node_type_transforms in node_feature_transforms.items():
        for feature_name, feature_transforms in node_type_transforms.items():
            assert (
                feature_transforms
                == reapplied_transformations["node_features"][node_type][feature_name]
            )

    # TODO: Verify other metadata files that verify_integ_test_output doesn't check for


def test_merge_config_and_feat_dim(tempdir: str, executor_configuration: ExecutorConfig):
    """Test the _merge_config_and_feat_dim function with hardcoded data"""
    dist_executor = DistributedExecutor(executor_configuration)

    # Mock the SparkContext stop() function to leave the Spark context running
    # for the other tests, otherwise dist_executor stops it
    dist_executor.spark.stop = mock.MagicMock(name="stop")

    dist_executor.run()
    with open(os.path.join(tempdir, "metadata.json"), "r", encoding="utf-8") as mfile:
        metadata = json.load(mfile)

    dist_executor._merge_config_with_feat_dim(dist_executor.gsp_config_dict, metadata)

    for node_dict_per_type in dist_executor.gsp_config_dict["nodes"]:
        node_type = node_dict_per_type["type"]
        if "features" in node_dict_per_type:
            nfeat_dict_per_type = NODE_CLASS_GRAPHINFO_UPDATES["nfeat_size"][node_type]
            for node_feat_dict in node_dict_per_type["features"]:
                feat_name = (
                    node_feat_dict["name"] if "name" in node_feat_dict else node_feat_dict["column"]
                )
                assert node_feat_dict["dim"] == [nfeat_dict_per_type[feat_name]]

    for edge_dict_per_type in dist_executor.gsp_config_dict["edges"]:
        src_type = edge_dict_per_type["source"]["type"]
        dst_type = edge_dict_per_type["dest"]["type"]
        relation = edge_dict_per_type["relation"]["type"]
        edge_type = f"{src_type}:{relation}:{dst_type}"
        if "features" in edge_dict_per_type:
            efeat_dict_per_type = NODE_CLASS_GRAPHINFO_UPDATES["efeat_size"][edge_type]
            for edge_feat_dict in edge_dict_per_type["features"]:
                feat_name = (
                    edge_feat_dict["name"] if "name" in edge_feat_dict else edge_feat_dict["column"]
                )
                assert edge_feat_dict["dim"] == [efeat_dict_per_type[feat_name]]


def test_merge_input_and_transform_dicts(executor_configuration: ExecutorConfig):
    """Test the _merge_config_with_transformations function with hardcoded json data"""
    dist_executor = DistributedExecutor(executor_configuration)

    pre_comp_transormations = {
        "node_features": {
            "user": {
                "state": {
                    "transformation_name": "categorical",
                }
            }
        },
        "edge_features": {},
    }

    input_config_with_transforms = dist_executor._merge_config_with_transformations(
        dist_executor.gsp_config_dict,
        pre_comp_transormations,
    )

    # Ensure the "user" node type's "state" feature includes a transformation entry
    for node_input_dict in input_config_with_transforms["graph"]["nodes"]:
        if "user" == node_input_dict["type"]:
            for feature in node_input_dict["features"]:
                if "state" == feature["column"]:
                    transform_for_feature = feature["precomputed_transformation"]
                    assert transform_for_feature["transformation_name"] == "categorical"


def test_dist_executor_graph_name(executor_configuration: ExecutorConfig):
    """Test cases for graph name"""

    # Ensure we can set a valid graph name
    executor_configuration.graph_name = "2024-a_valid_name"
    dist_executor = DistributedExecutor(executor_configuration)
    assert dist_executor.graph_name == "2024-a_valid_name"

    # Ensure default value is used when graph_name is not provided
    executor_configuration.graph_name = None
    dist_executor = DistributedExecutor(executor_configuration)
    assert dist_executor.graph_name == "small_heterogeneous_graph"

    # Ensure we raise when invalid graph name is provided
    with pytest.raises(AssertionError):
        executor_configuration.graph_name = "graph.name"
        dist_executor = DistributedExecutor(executor_configuration)

    # Ensure a valid default graph name is parsed when the input ends in '/'
    executor_configuration.graph_name = None
    executor_configuration.input_prefix = executor_configuration.input_prefix + "/"
    dist_executor = DistributedExecutor(executor_configuration)
    assert dist_executor.graph_name == "small_heterogeneous_graph"
