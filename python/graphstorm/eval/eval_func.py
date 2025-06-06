"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Evaluation functions
"""
import logging
import operator
from collections.abc import Callable
from enum import Enum
from functools import partial

import numpy as np
import torch as th
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (precision_recall_curve,
                             auc,
                             classification_report,
                             precision_recall_fscore_support)
from .utils import is_float


SUPPORTED_RECALL_AT_PRECISION_METRICS = 'recall_at_precision'
SUPPORTED_PRECISION_AT_RECALL_METRICS = 'precision_at_recall'
SUPPORTED_FSCORE_AT_METRICS = 'fscore_at'
SUPPORTED_HIT_AT_METRICS = 'hit_at'
SUPPORTED_CLASSIFICATION_METRICS = {'accuracy', 'precision_recall', \
    'roc_auc', 'f1_score', 'per_class_f1_score', 'per_class_roc_auc', 'precision', 'recall', \
    SUPPORTED_HIT_AT_METRICS, SUPPORTED_FSCORE_AT_METRICS, \
    SUPPORTED_RECALL_AT_PRECISION_METRICS, SUPPORTED_PRECISION_AT_RECALL_METRICS}
SUPPORTED_REGRESSION_METRICS = {'rmse', 'mse', 'mae'}
SUPPORTED_LINK_PREDICTION_METRICS = {"mrr", SUPPORTED_HIT_AT_METRICS, "amri"}

class ClassificationMetrics:
    """ object that compute metrics for classification tasks.
    
    Note(Jian): In order to let users to implement their own metrics, we need to:
    1) refactorize this Metrics class to expose the comparator, function, and eval_function
       interfaces to let users to set and get these objects.
    2) define a new MetricInterface class, and define two abs methods, i.e.,
        - assert_supported_metric;
        - init_best_metric.
    3) further discuss if we can set all metrics (Classsification, Regression, and LP) in the same
       architecture.
    """
    def __init__(self, eval_metric_list, multilabel):
        self.supported_metrics = SUPPORTED_CLASSIFICATION_METRICS
        self.multilabel = multilabel

        # This is the operator used to compare whether current value is better than the current best
        self.metric_comparator = {}
        self.metric_comparator["accuracy"] = operator.le
        self.metric_comparator["precision_recall"] = operator.le
        self.metric_comparator["roc_auc"] = operator.le
        self.metric_comparator["f1_score"] = operator.le
        self.metric_comparator["per_class_f1_score"] = comparator_per_class_f1_score
        self.metric_comparator["per_class_roc_auc"] = comparator_per_class_roc_auc
        self.metric_comparator["precision"] = operator.le
        self.metric_comparator["recall"] = operator.le

        # This is the operator used to measure each metric performance in training
        self.metric_function = {}
        self.metric_function["accuracy"] = partial(compute_acc, multilabel=self.multilabel)
        self.metric_function["precision_recall"] = compute_precision_recall_auc
        self.metric_function["roc_auc"] = compute_roc_auc
        self.metric_function["f1_score"] = compute_f1_score
        self.metric_function["per_class_f1_score"] = compute_f1_score
        self.metric_function["per_class_roc_auc"] = compute_roc_auc
        self.metric_function["precision"] = compute_precision
        self.metric_function["recall"] = compute_recall

        # This is the operator used to measure each metric performance in evaluation
        self.metric_eval_function = {}
        self.metric_eval_function["accuracy"] = partial(compute_acc, multilabel=self.multilabel)
        self.metric_eval_function["precision_recall"] = compute_precision_recall_auc
        self.metric_eval_function["roc_auc"] = compute_roc_auc
        self.metric_eval_function["f1_score"] = compute_f1_score
        self.metric_eval_function["per_class_f1_score"] = compute_per_class_f1_score
        self.metric_eval_function["per_class_roc_auc"] = compute_per_class_roc_auc
        self.metric_eval_function["precision"] = compute_precision
        self.metric_eval_function["recall"] = compute_recall

        for eval_metric in eval_metric_list:
            if eval_metric.startswith(SUPPORTED_HIT_AT_METRICS):
                k = int(eval_metric[len(SUPPORTED_HIT_AT_METRICS)+1:])
                self.metric_comparator[eval_metric] = operator.le
                self.metric_function[eval_metric] = \
                    partial(compute_hit_at_classification, k=k)
                self.metric_eval_function[eval_metric] = \
                    partial(compute_hit_at_classification, k=k)

            if eval_metric.startswith(SUPPORTED_FSCORE_AT_METRICS):
                beta = float(eval_metric[len(SUPPORTED_FSCORE_AT_METRICS)+1:].strip())
                self.metric_comparator[eval_metric] = operator.le
                self.metric_function[eval_metric] = partial(compute_fscore, beta=beta)
                self.metric_eval_function[eval_metric] = partial(compute_fscore, beta=beta)

            if eval_metric.startswith(SUPPORTED_PRECISION_AT_RECALL_METRICS):
                beta = float(eval_metric[len(SUPPORTED_PRECISION_AT_RECALL_METRICS)+1:].strip())
                self.metric_comparator[eval_metric] = operator.le
                self.metric_function[eval_metric] = partial(compute_precision_at_recall, beta=beta)
                self.metric_eval_function[eval_metric] = partial(compute_precision_at_recall,
                                                                 beta=beta)

            if eval_metric.startswith(SUPPORTED_RECALL_AT_PRECISION_METRICS):
                beta = float(eval_metric[len(SUPPORTED_RECALL_AT_PRECISION_METRICS)+1:].strip())
                self.metric_comparator[eval_metric] = operator.le
                self.metric_function[eval_metric] = partial(compute_recall_at_precision, beta=beta)
                self.metric_eval_function[eval_metric] = partial(compute_recall_at_precision,
                                                                 beta=beta)

    def assert_supported_metric(self, metric):
        """ check if the given metric is supported.
        """
        if metric.startswith(SUPPORTED_HIT_AT_METRICS):
            assert metric[len(SUPPORTED_HIT_AT_METRICS)+1:].isdigit(), \
                            "hit_at_k evaluation metric for classification " \
                            f"must end with an integer, but get {metric}"
        elif metric.startswith(SUPPORTED_FSCORE_AT_METRICS):
            assert is_float(metric[len(SUPPORTED_FSCORE_AT_METRICS)+1:]), \
                            "fscore_at_beta evaluation metric for classification " \
                            f"must end with an integer or float, but get {metric}"
        elif metric.startswith(SUPPORTED_PRECISION_AT_RECALL_METRICS):
            assert is_float(metric[len(SUPPORTED_PRECISION_AT_RECALL_METRICS)+1:]), \
                            "precision_at_recall evaluation metric for classification " \
                            f"must end with an integer or float, but get {metric}"
            assert 0 < float(metric[len(SUPPORTED_PRECISION_AT_RECALL_METRICS)+1:]) <= 1, \
                "The beta in precision_at_recall evaluation metric must be in (0, 1], " \
                f"but get {float(metric[len(SUPPORTED_PRECISION_AT_RECALL_METRICS)+1:])}."
        elif metric.startswith(SUPPORTED_RECALL_AT_PRECISION_METRICS):
            assert is_float(metric[len(SUPPORTED_RECALL_AT_PRECISION_METRICS)+1:]), \
                            "recall_at_precision evaluation metric for classification " \
                            f"must end with an integer or float, but get {metric}"
            assert 0 < float(metric[len(SUPPORTED_RECALL_AT_PRECISION_METRICS)+1:]) <= 1, \
                "The beta in recall_at_precision evaluation metric must be in (0, 1], " \
                f"but get {float(metric[len(SUPPORTED_RECALL_AT_PRECISION_METRICS)+1:])}."
        else:
            assert metric in self.supported_metrics, \
                f"Metric {metric} not supported for classification"

    def init_best_metric(self, metric):
        """
        Return the initial value for the metric to keep track of the best metric.
        Parameters
        ----------
        metric: the metric to initialize

        Returns
        -------

        """
        # Need to check if the given metric is supported first
        self.assert_supported_metric(metric)
        return 0


class RegressionMetrics:
    """ object that compute metrics for regression tasks.
    """
    def __init__(self):
        self.supported_metrics = SUPPORTED_REGRESSION_METRICS

        # This is the operator used to compare whether current value is better than the current best
        self.metric_comparator = {}
        self.metric_comparator["rmse"] = operator.ge
        self.metric_comparator["mse"] = operator.ge
        self.metric_comparator["mae"] = operator.ge

        # This is the operator used to measure each metric performance
        self.metric_function = {}
        self.metric_function["rmse"] = compute_rmse
        self.metric_function["mse"] = compute_mse
        self.metric_function["mae"] = compute_mae

        # This is the operator used to measure each metric performance in evaluation
        self.metric_eval_function = {}
        self.metric_eval_function["rmse"] = compute_rmse
        self.metric_eval_function["mse"] = compute_mse
        self.metric_eval_function["mae"] = compute_mae

    def assert_supported_metric(self, metric):
        """ check if the given metric is supported.
        """
        assert metric in self.supported_metrics, \
            f"Metric {metric} not supported for regression"

    def init_best_metric(self, metric):
        """
        Return the initial value for the metric to keep track of the best metric.
        Parameters
        ----------
        metric: the metric to initialize

        Returns
        -------

        """
        # Need to check if the given metric is supported first
        self.assert_supported_metric(metric)
        return np.finfo(np.float32).max


class LinkPredictionMetrics:
    """ object that compute metrics for LP tasks.

    Parameters
    ----------
    eval_metric_list: list of string
        Evaluation metric(s) used during evaluation, for example, ["mrr", "hit_at_1", "hit_at_100"].
    """
    def __init__(self, eval_metric_list=None):
        self.supported_metrics = SUPPORTED_LINK_PREDICTION_METRICS

        # This is the operator used to compare whether current value is better than the current best
        self.metric_comparator: dict[str, Callable] = {}
        self.metric_comparator["mrr"] = operator.le
        self.metric_comparator["amri"] = operator.le

        # This is the operator used to measure each metric performance
        self.metric_function: dict[str, Callable[..., th.Tensor]] = {}
        self.metric_function["mrr"] = compute_mrr
        self.metric_function["amri"] = compute_amri

        # This is the operator used to measure each metric performance in evaluation
        self.metric_eval_function: dict[str, Callable[..., th.Tensor]] = {}
        self.metric_eval_function["mrr"] = compute_mrr
        self.metric_eval_function["amri"] = compute_amri

        if eval_metric_list:
            for eval_metric in eval_metric_list:
                if eval_metric.startswith(SUPPORTED_HIT_AT_METRICS):
                    k = int(eval_metric[len(SUPPORTED_HIT_AT_METRICS) + 1:])
                    self.metric_comparator[eval_metric] = operator.le
                    self.metric_function[eval_metric] = \
                        partial(compute_hit_at_link_prediction, k=k)
                    self.metric_eval_function[eval_metric] = \
                        partial(compute_hit_at_link_prediction, k=k)

    def assert_supported_metric(self, metric):
        """ check if the given metric is supported.
        """
        if metric.startswith(SUPPORTED_HIT_AT_METRICS):
            assert metric[len(SUPPORTED_HIT_AT_METRICS) + 1:].isdigit(), \
                "hit_at_k evaluation metric for link prediction " \
                f"must end with an integer, but get {metric}"
        else:
            assert metric in self.supported_metrics, \
                f"Metric {metric} not supported for link prediction"

    def init_best_metric(self, metric: str):
        """
        Return the initial value for the metric to keep track of the best metric.
        Parameters
        ----------
        metric: str
            the name of the metric to initialize

        Returns
        -------
        float
            An initial value for the metric.
        """
        # Need to check if the given metric is supported first
        self.assert_supported_metric(metric)
        # The minimum value for AMRI is -1.0 so we init with that
        return -1.0 if metric == "amri" else 0.0


def labels_to_one_hot(labels, total_labels):
    '''
    This function converts the original labels to an one hot array.

    Parameters
    ----------
    labels: 1D list
        The label list.

    total_labels: int
        Number of unique labels.

    Returns
    -------
    np.array: One-hot encoding of the labels in the format of N * total_labels.
    '''

    if len(labels.shape)>1:
        return labels
    one_hot=np.zeros(shape=(len(labels),total_labels))
    for i, label in enumerate(labels):
        one_hot[i,label]=1
    return one_hot

def compute_hit_at_classification(preds, labels, k=100):
    """ Compute hit@k for classification tasks

        Parameters
        ----------
        preds : tensor
            A 1-D tensor for single-label classification.
        labels : tensor
            A 1-D tensor for single-label classification.
        k: int
            Hit@K
    """
    assert len(preds.shape) == 2 \
        and preds.shape[1] <= 2, \
        "Computing hit@K for classification only works for binary classification tasks." \
        "The preds must be a 2D tensor with the second dimension of 1 or 2. "

    assert len(labels.shape) == 1 or (len(labels.shape) == 2 and labels.shape[1] == 1), \
        "The labels must be a 1D tensor or a 2D tensor with the second dimension of 1"

    # preds is a 2D tensor storing
    # [probability of label 0, probability of label 1]
    # 0 means negative, 1 means positive.
    # We compute hit@K for positive labels
    preds = preds[:,1] if preds.shape[1] == 2 else preds.squeeze()
    if len(labels.shape) == 2:
        labels = th.squeeze(labels)
    sort_idx = th.argsort(preds, descending=True)
    hit_idx = sort_idx[:k]
    hit_labels = labels[hit_idx]
    return th.sum(hit_labels)


def compute_hit_at_link_prediction(ranking, k=100):
    """ Compute hit@k for link prediction tasks

        Parameters
        ----------
        ranking: tensor
            A tensor for the ranking of positive edges
        k: int
            Hit@K

        Returns
        -------
        float: Hit at K score
    """
    assert len(ranking.shape) == 1 or (len(ranking.shape) == 2 and ranking.shape[1] == 1), \
        "The ranking must be a 1D tensor or a 2D tensor with the second dimension of 1. "

    if len(ranking.shape) == 2:
        ranking = th.squeeze(ranking)

    metric = th.div(th.sum(ranking <= k), len(ranking))
    return metric

def eval_roc_auc(logits,labels):
    ''' Compute roc_auc score.
        If any errors occur, raise the error to callers and stop.

        Parameters
        ----------
        logits : Target scores in 2D tensor.
                Array-like of shape (n_samples, n_classes) with logits.
        labels: Array-like of shape (n_samples,) or (n_samples, n_classes) True labels or
                binary label indicators. The binary and multiclass cases expect labels with
                shape (n_samples,) while the multilabel case expects binary label indicators
                with shape (n_samples, n_classes).

        Returns
        -------
        float: The roc_auc score.
    '''
    predicted_labels=logits
    predicted_labels=predicted_labels.detach().cpu().numpy()
    labels=labels.detach().cpu().numpy()

    # check if the two inputs have the same number of rows.
    assert predicted_labels.shape[0] == labels.shape[0], 'ERROR: Predictions and labes ' + \
                f'should have the same number of records, but got, {predicted_labels.shape[0]}' + \
                f' and {labels.shape[0]}.'

    # check if the predictions is 2D.
    assert len(predicted_labels.shape) == 2, 'ERROR: GraphStorm assumes the predicted ' + \
                                             'logit is a 2D tesnor, but got a 1D tensor.'

    # The roc_auc_score function computes the area under the receiver operating characteristic
    # (ROC) curve, which is also denoted by AUC or AUROC. The following returns the average AUC.
    rocauc_list = []

    # Binary results, the sklearn roc_auc_score function asks 1D inputs of predictions.
    # And the label is a 1D tensor. For other cases, the sklearn roc_auc_score function asks
    # nD inputs of predictions. So here we need to check the predictions' 2nd dim for the binary
    # conditions.
    if predicted_labels.shape[1] == 2:
        if len(labels.shape) == 1:
            # Here use the 2nd dim, assuming it is the probability of 1s.
            rocauc_list.append(roc_auc_score(labels, predicted_labels[:, 1]))
            return sum(rocauc_list) / len(rocauc_list)
        elif len(labels.shape) == 2 and labels.shape[1] == 1:
            # Here use the 2nd dim, assuming it is the probability of 1s.
            rocauc_list.append(roc_auc_score(labels.squeeze(), predicted_labels[:, 1]))
            return sum(rocauc_list) / len(rocauc_list)

    # mutiple class and multiple labels cases
    try:
        labels=labels_to_one_hot(labels, predicted_labels.shape[1])
    except IndexError as e:
        logging.error("Failure found during evaluation of the roc_auc score metric due to" + \
                      " reason: %s", str(e))
        raise

    for i in range(labels.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(labels[:, i] == 1) > 0 and np.sum(labels[:, i] == 0) > 0:
            is_labeled = labels[:, i] == labels[:, i]
            rocauc_list.append(roc_auc_score(labels[is_labeled, i],
                                            predicted_labels[is_labeled, i]))

    if len(rocauc_list) == 0:
        logging.error('No positively labeled data available. Cannot compute ROC-AUC.')
        return 0

    return sum(rocauc_list) / len(rocauc_list)

def eval_acc(pred, labels):
    """compute evaluation accuracy.
       If any errors occur, raise the error to callers and stop.

        Parameters
        ----------
        pred : 1D or 2D tensor.
            Target scores in 1D tensor of 0s and 1s, or 2D tensor of logits.
        labels: 1D tensor.
            Labels in 1D tensor of 0s and 1s.
        Returns
        -------
        float: The accuracy score.
    """
    if pred.dim() > 1:
        if pred.shape[1] == 1:
            pred = th.squeeze(pred)
        else:
            # if pred has dimension > 1, and the second dim > 1,
            # it has full logits instead of final prediction
            assert th.is_floating_point(pred), \
                "ERROR: Multiple dimension logits are expected to " + \
                f"be float type. But get {pred.dtype}"
            pred = pred.argmax(dim=1)
    # Check if pred is integer tensor
    assert (not th.is_floating_point(pred) and not th.is_complex(pred)), "ERROR: 1D " + \
                                                                    "predictions are " + \
                                                                    "expected to be integer type."

    return th.sum(pred.cpu() == labels.cpu()).item() / len(labels)

def compute_f1_score(y_preds, y_targets):
    """ compute macro_average f1 score.
        If any errors occur, raise the error to callers and stop.

        Parameters
        ----------
        y_preds : 1D list of 0s and 1s
            predictions after argmax.
        y_targets : 1D list of 0s and 1s
            The 1D label list.

        Returns
        -------
        float: The f1 score.
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    try:
        report = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
        f1_score = report['macro avg']['f1-score']
    except ValueError as e:
        logging.error("Failure found during evaluation of the f1 score metric due to" + \
                      " reason: %s", str(e))
        raise

    return f1_score

def compute_per_class_f1_score(y_preds, y_targets):
    """ compute f1 score per class
        If any errors occur, raise the error to callers and stop.

        Parameters
        ----------
        y_preds : 1D list of 0s and 1s
            predictions after argmax.
        y_targets : 1D list of 0s and 1s
            The 1D label list.

        Returns
        -------
        float: The f1 score.
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    try:
        report = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
    except ValueError as e:
        logging.error("Failure found during evaluation of the per class f1 score metric due to" + \
                      "reason: %s", str(e))
        raise

    return report

def comparator_per_class_f1_score(best_report, current_report):
    """ compare method for f1 score per class
    """
    return best_report['macro avg']['f1-score'] < current_report['macro avg']['f1-score']\
        if best_report != 0 else 0 < current_report['macro avg']['f1-score']

def compute_acc_lp(pos_score, neg_score):
    """
    This function calculates the LP accuracy. It is a cheap and fast way to evaluate the
    accuracy of the model. The scores are ranked from larger to smaller. If all the pos_scores
    are ranked before all the neg_scores then the value returned is 1 that is the maximum.

    Parameters
    ----------
    pos_score : the positive scores
    neg_score : the negative scores

    Returns
    -------
    lp_score : the lp accuracy.

    """
    num_pos=len(pos_score)
    # perturb object
    scores = th.cat([pos_score, neg_score], dim=0)
    scores = th.sigmoid(scores)
    _, rankings = th.sort(scores, dim=0, descending=True)
    rankings = rankings.cpu().detach().numpy()
    rankings = rankings <= num_pos
    lp_score = sum(rankings[:num_pos]) / num_pos

    return {"lp_fast_score": lp_score}

def compute_roc_auc(y_preds, y_targets, weights=None):
    """ compute ROC's auc score with weights
        If any errors occur, raise the error to callers and stop.

        Parameters
        ----------
        y_preds : Target scores in 2D tensor.
                  Array-like of shape (n_samples, n_classes) with logits.
        y_targets: Array-like of shape (n_samples,) or (n_samples, n_classes) True labels or
                   binary label indicators. The binary and multiclass cases expect labels with
                   shape (n_samples,) while the multilabel case expects binary label indicators
                   with shape (n_samples, n_classes).
        weights: List of weights with the same number of classes in labels.
        Returns
        -------
        float: The roc_auc score.
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()

    # check for binary cases, input (n, 2) and label 1D or (n, 1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
        if len(y_true.shape) == 1:
            y_pred = y_pred[:, 1]
        elif len(y_true.shape) == 2 and y_true.shape[1] == 1:
            y_pred = y_pred[:, 1]
            y_true = y_true.squeeze()

    if weights is not None:
        weights = weights.cpu().numpy()

    # adding checks since in certain cases the auc might not be defined we do not want to fail
    # the code
    try:
        auc_score = roc_auc_score(y_true, y_pred, sample_weight=weights, multi_class='ovr')
    except ValueError as e:
        logging.error("Failure found during evaluation of the roc_auc metric due to the" + \
                      " reason: %s", str(e))
        raise

    return auc_score

def comparator_per_class_roc_auc(best_report, current_report):
    """ compare method for roc_auc score per class
    """
    return best_report["overall avg"] < current_report["overall avg"] \
        if best_report != 0 else 0 < current_report["overall avg"]

def compute_per_class_roc_auc(y_preds, y_targets):
    """ compute ROC-AUC score per class
        If any errors occur, raise the error to callers and stop.

        Parameters
        ----------
        y_preds : Target scores in 2D tensor with shape (n_samples, n_classes).
        y_targets: the multilple classes case with shape (n_samples, n_classes).

        The number of classes of y_preds should be equal to the number of classes of y_targets.
        Returns
        -------
        A dictionary of auc_roc scores, including average auc_roc score, and score for each class.

    """
    assert len(y_preds.shape) == 2 and y_preds.shape[1] >= 2, 'ERROR: the given prediction ' + \
                                                              'should be a 2D tensor and the ' + \
                                                              '2nd dimension should be >= 2, ' + \
                                                              f'but got {y_preds.shape}.'

    assert len(y_targets.shape) == 2 and y_targets.shape[1] >= 2, 'ERROR: the given labels ' + \
                                                              'should be a 2D tensor and the ' + \
                                                              '2nd dimension should be >= 2, ' + \
                                                              f'but got {y_targets.shape}.'

    assert y_preds.shape[1] == y_targets.shape[1], 'ERROR: the 2nd dimension of predictions ' + \
                                                   'and labels should be the same, but got ' + \
                                                   f'{y_preds.shape} and {y_targets.shape}.'

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    roc_auc_report = {}

    avg_roc_auc = compute_roc_auc(y_preds, y_targets)
    for class_id in range(y_true.shape[1]):
        try:
            roc_auc_report[class_id] = roc_auc_score(y_true[:,class_id],
                                                    y_pred[:,class_id])
        except ValueError as e:
            logging.error("Failure found during evaluation of the roc_auc_score metric due to " + \
                          "the reason: %s", str(e))
            raise
    roc_auc_report["overall avg"] = avg_roc_auc

    return roc_auc_report


class PRKeys(str, Enum):
    """ Enums support iteration in definition order--order matters here
    """
    PRECISION = "precision"
    RECALL = "recall"
    THRESHOLD = "threshold"
    FSCORE = "fscore"


def compute_precision_recall_auc(y_preds, y_targets, weights=None):
    """ compute precision, recall, and auc values.
         If any errors occur, raise the error to callers and stop.

        Parameters
        ----------
        y_preds : Target scores in 2D tensor.
        y_targets: Array-like of shape (n_samples,) or (n_samples, n_classes) True labels or
                   binary label indicators. The binary and multiclass cases expect labels with
                   shape (n_samples,) while the multilabel case expects binary label indicators
                   with shape (n_samples, n_classes).
        weights: List of weights with the same number of classes in labels.
        Returns
        -------
        float: The precision_recall_auc score.
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()

    # same check for binary cases, input in (n, 2) and label in 1D or (n, 1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
        if len(y_true.shape) == 1:
            y_pred = y_pred[:, 1]
        elif len(y_true.shape) == 2 and y_true.shape[1] == 1:
            y_pred = y_pred[:, 1]
            y_true = y_true.squeeze()

    keys = [key.value for key in PRKeys]

    # adding checks since in certain cases the auc might not be defined we do not want to fail
    # the code
    try:
        pr_curve = dict(zip(keys, precision_recall_curve(y_true, y_pred, sample_weight=weights)))
        precision, recall = pr_curve[PRKeys.PRECISION], pr_curve[PRKeys.RECALL]
        auc_score = auc(recall, precision)
    except ValueError as e:
        logging.error("Failure found during evaluation of the precision_recall_auc metric due " + \
                      "to reason: %s", str(e))
        raise

    return auc_score

def compute_precision_recall_fscore(y_preds, y_targets, beta=2.):
    """ Compute Precision, Recall, and Fscore
    
    In order to provide a sigle-value evaluation, for binary classification, it will return a
    binary value. For multi-class classification, it will return a macro-averaged value.
    For multi-label cases, it will return a list of values, one for each label. It is up to the
    caller to decide how to handle the multi-label values.

    Details can be found in
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

    Parameters
    ----------
    pred : tensor
        a 1-D tensor for single-label classification and 2-D tensor for multi-label classification.
        For 2-D tensor, the number of column is the number of labels.
    labels : tensor
        a 1-D tensor for single-label classification and 2-D tensor for multi-label classification.
        For 2-D tensor, the number of column is the number of labels.
    beta : float
        The beta value for computing fscore. Default is 2.0.

    Returns
    -------
    precision: A float value, or list of values of precision for multi-label.
    recall: A float value, or list of values of recall for multi-label.
    fscore: A float value, or list of values of fscore for multi-label.
    """
    # check prediction values. Must be integers, not float logits.
    assert not (th.is_floating_point(y_preds) or th.is_complex(y_preds)), 'The predictions ' + \
                                            f'should be integer values, but got {y_preds.dtype}.'

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()

    if len(y_pred.shape) == 1:   # 1-D tensor for single-label classification, using macro avg
        assert len(y_true.shape) == 1, 'The provided labels should be 1D' + \
                                       ' for single-label classification.'
        if y_pred.max() == 1 and y_true.max() == 1:     # 1-D binary tensor, using binary
            precision, recall, fscore, _ = precision_recall_fscore_support(y_pred=y_true,
                                                                           y_true=y_pred,
                                                                           beta=beta,
                                                                           average='binary'
                                                                           )
        else:
            precision, recall, fscore, _ = precision_recall_fscore_support(y_pred=y_true,
                                                                           y_true=y_pred,
                                                                           beta=beta,
                                                                           average='macro'
                                                                           )
    elif len(y_pred.shape) == 2:   # 2-D tensor for multi-label classification, returning per class
        assert len(y_true.shape) == 2, 'The provided labels should be 2D for multi-label ' + \
                                       f'classification, but got label shape: {y_true.shape}'
        precision, recall, fscore, _ = precision_recall_fscore_support(y_pred=y_true,
                                                                   y_true=y_pred,
                                                                   beta=beta)
    else:
        raise NotImplementedError(f'Not support >2D predictions, but got shape: {y_pred.shape}')

    return precision, recall, fscore

def compute_precision(y_preds, y_targets):
    """ Compute precision for classification tasks
    """
    precision, _, _ = compute_precision_recall_fscore(y_preds, y_targets)

    return precision

def compute_recall(y_preds, y_targets):
    """ Compute recall for classification tasks
    """
    _, recall, _ = compute_precision_recall_fscore(y_preds, y_targets)

    return recall

def compute_fscore(y_preds, y_targets, beta):
    """ Compute fscore for classification tasks
    """
    _, _, fscore = compute_precision_recall_fscore(y_preds, y_targets, beta)

    return fscore

def compute_precision_at_recall(y_preds, y_targets, beta=1., weights=None):
    """ Compute precision at recall at beta for binary classification tasks.
        If there is no a recall score equal to beta, it returns the precision
            at the largest recall less than beta.
        If there are multiple precision scores when recall equal to beta
            or the largest recall less than beta, it returns the maximum precision
            among these precision scores.
        If unable to find a proper precision with the given beta value,
            it will return 0 as precision, and provide warning message.

        This metric is only for binary classification tasks.

        Parameters
        ----------
        y_preds : tensor
            Target scores in 1D or 2D tensor. Tensors with more than 2D will trigger an error.
        y_targets: tensor
            Array-like of shape (n_samples,) or (n_samples, n_classes) True labels or
            binary label indicators.
        weights: list
            A list of weights with the same number of classes in labels.
            Default is None.
        beta: float or int
            Beta value of precision for getting recall. Should be in the range of (0, 1].
            Default is 1.0.

        Returns
        -------
        float: The precision_at_recall score.
    """
    assert 0 < beta <= 1, f"ERROR: beta should be in the range of (0, 1], but get {beta}"

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()

    # only support binary classification
    nclass = len(np.unique(y_true))
    assert nclass <= 2, (f"ERROR: compute_precision_at_recall only supports binary "
                         f"classification, but got {nclass} classes.")

    # same check for binary cases, input in (n, 2) and label in 1D or (n, 1)
    assert len(y_pred.shape) <= 2, (f"ERROR: not support >2D predictions, "
                                    f"but got shape: {y_pred.shape}")
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
        if len(y_true.shape) == 1:
            y_pred = y_pred[:, 1]
        elif len(y_true.shape) == 2 and y_true.shape[1] == 1:
            y_pred = y_pred[:, 1]
            y_true = y_true.squeeze()

    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred,
                                                  sample_weight=weights)

    if beta in recall:
        locations = np.where(recall == beta)[0]
        return np.max(precision[locations])
    else:
        # sort the recall and precision lists by the ascending order of recall
        sort_idx = np.argsort(recall)
        recall_sorted_asc = recall[sort_idx]
        precision_sorted = precision[sort_idx]

        idx = np.searchsorted(recall_sorted_asc, beta) - 1
        if idx < 0:
            logging.warning(
                "WARNING: could not find a corresponding precision score given beta %s. "
                "Return 0 for precision@recall instead.", beta)
            return 0.

        new_beta = recall_sorted_asc[idx]
        locations = np.where(recall_sorted_asc == new_beta)[0]
        return np.max(precision_sorted[locations])

def compute_recall_at_precision(y_preds, y_targets, beta=1., weights=None):
    """ Compute recall at precision at beta for binary classification tasks.
        If there is no precision score equal to beta, we update the beta as the first precision
            less than beta following the ascending order of corresponding recall. For example, given
            a list of recall [0., 0.5, 0.5, 1., 1.], a list of precision [1., 1., 0.5, 0.67, 0.5],
            and beta=0.6, the updated beta will be 0.5 whose index is 2 in the precision list.
        With the known index of the first met precision equal to beta or the updated beta in the
            precision list (following the ascending order of corresponding recall list), to find the
            location of the last met maximum precision in the slice of precision list
            from the known index to the end, it will return the recall at that location.
            For example, the known index is 2 as above, the location of the maximum precision in the
            slice of precision list is 3, and the returned recall will be 1. at the 3rd index of the
            recall list.
        If unable to find a proper precision with the given beta value,
            it will return 0 as precision, and provide warning message.

        This metric is only for binary classification tasks.

        Parameters
        ----------
        y_preds : tensor
            Target scores in 1D or 2D tensor. Tensors with more than 2D will trigger an error.
        y_targets: tensor
            Array-like of shape (n_samples,) or (n_samples, n_classes) True labels or
            binary label indicators.
        weights: list
            A list of weights with the same number of classes in labels.
            Default is None.
        beta: float or int
            Beta value of precision for getting recall. Should be in the range of (0, 1].
            Default is 1.0.

        Returns
        -------
        float: The precision_at_recall score.
    """
    assert 0 < beta <= 1, f"ERROR: beta should be in the range of (0, 1], but get {beta}"

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()

    # only support binary classification
    nclass = len(np.unique(y_true))
    assert nclass <= 2, (f"ERROR: compute_precision_at_recall only supports binary "
                                    f"classification, but got {nclass} classes.")

    # same check for binary cases, input in (n, 2) and label in 1D or (n, 1)
    assert len(y_pred.shape) <= 2, (f"ERROR: not support >2D predictions, "
                                    f"but got shape: {y_pred.shape}")
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 2:
        if len(y_true.shape) == 1:
            y_pred = y_pred[:, 1]
        elif len(y_true.shape) == 2 and y_true.shape[1] == 1:
            y_pred = y_pred[:, 1]
            y_true = y_true.squeeze()

    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred,
                                                  sample_weight=weights)

    if beta in precision:
        locations = np.where(precision == beta)[0]
        return np.max(recall[locations])
    else:
        # sort the recall and precision lists by the ascending order of recall
        sort_idx = np.argsort(recall)
        recall_sorted_asc = recall[sort_idx]
        precision_sorted = precision[sort_idx]

        new_beta = None
        for prec in precision_sorted:
            if prec < beta:
                new_beta = prec
                break
        if new_beta is None:
            logging.warning(
                "WARNING: could not find a corresponding recall score given beta %s. "
                "Return 0 for recall@precision instead.", beta)
            return 0.

        # returns the maximum recall at precision == new_beta
        return np.max(recall_sorted_asc[precision_sorted == new_beta])

def compute_acc(pred, labels, multilabel):
    '''Compute accuracy.

    Parameters
    ----------
    pred : tensor
        a 1-D tensor for single-label classification and 2-D tensor for multi-label classification.
        For 2-D tensor, the number of column is the number of labels.
    labels : tensor
        a 1-D tensor for single-label classification and 2-D tensor for multi-label classification.
        For 2-D tensor, the number of column is the number of labels.
    multilabel : bool
        Whether this is a multi-label classification task.

    Returns
    -------
        A 1-D tensor that stores the accuracy.
    '''
    if multilabel:
        return eval_roc_auc(pred, labels)
    else:
        return eval_acc(pred, labels)

def compute_rmse(pred, labels):
    """ compute RMSE for regression.
    """
    # TODO: check dtype of label before training or evaluation
    assert th.is_floating_point(pred) and th.is_floating_point(labels), \
        "prediction and labels must be floating points"

    # Handle the case when the label is a 1D tensor and
    # the prediction result has the shape as (len(labels), 1)
    if len(labels.shape) == 1 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    assert pred.shape == labels.shape, \
        f"prediction and labels have different shapes. {pred.shape} vs. {labels.shape}"
    if pred.dtype != labels.dtype:
        logging.warning("prediction and labels have different data types: %s vs. %s.",
                        str(pred.dtype), str(labels.dtype))
        logging.warning("casting pred to the same dtype as labels.")
        pred = pred.type(labels.dtype) # cast pred to the same dtype as labels.

    diff = pred.cpu() - labels.cpu()
    return th.sqrt(th.mean(diff * diff)).cpu().item()

def compute_mse(pred, labels):
    """ compute MSE for regression
    """
    # TODO: check dtype of label before training or evaluation
    assert th.is_floating_point(pred) and th.is_floating_point(labels), \
        "prediction and labels must be floating points"

    # Handle the case when the label is a 1D tensor and
    # the prediction result has the shape as (len(labels), 1)
    if len(labels.shape) == 1 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    assert pred.shape == labels.shape, \
        f"prediction and labels have different shapes. {pred.shape} vs. {labels.shape}"
    if pred.dtype != labels.dtype:
        logging.warning("prediction and labels have different data types: %s vs. %s.",
                        str(pred.dtype), str(labels.dtype))
        logging.warning("casting pred to the same dtype as labels.")
        pred = pred.type(labels.dtype) # cast pred to the same dtype as labels.

    diff = pred.cpu() - labels.cpu()
    return th.mean(diff * diff).cpu().item()

def compute_mae(pred, labels):
    """ compute MAE for regression
    """
    # TODO: check dtype of label before training or evaluation
    assert th.is_floating_point(pred) and th.is_floating_point(labels), \
        "prediction and labels must be floating points"

    assert pred.shape == labels.shape, \
        f"prediction and labels have different shapes. {pred.shape} vs. {labels.shape}"
    if pred.dtype != labels.dtype:
        logging.warning("prediction and labels have different data types: %s vs. %s.",
                        pred.dtype, labels.dtype)
        logging.warning("casting pred to the same dtype as labels.")
        pred = pred.type(labels.dtype) # cast pred to the same dtype as labels.

    diff = th.abs(pred.cpu() - labels.cpu())
    return th.mean(diff).cpu().item()

def compute_mrr(ranking: th.Tensor) -> th.Tensor:
    """ Get link prediction Mean Reciprocal Rank (MRR) metrics

    Parameters
    ----------
    ranking: torch.Tensor
        ranking of each positive edge

    Returns
    -------
    th.Tensor
        link prediction mrr metrics
    """
    reciprocal_ranks = th.div(1.0, ranking)
    metrics = th.tensor(th.div(th.sum(reciprocal_ranks), len(reciprocal_ranks)))
    return metrics

def compute_amri(ranking: th.Tensor, candidate_sizes: th.Tensor) -> th.Tensor:
    """Computes the Adjusted Mean Rank Index (AMRI) for the given ranking and candidate sizes.

    AMRI is a metric that evaluates the performance of link prediction models by considering both
    the rank of the correct candidate and the number of candidates. It is calculated as:

    .. math::
        AMRI = 1 - \\frac{\\text{MR}-1}{\\mathbb{E}[\\text{MR}-1]}

    where MR is the mean rank, and `E[MR]` is the expected mean rank, which is used
    to adjust for chance. E[MR] is defined as:

    .. math::
        \\mathbb{E}[\\text{MR}] = \\mathbb{E} \\left[ \\frac{1}{n} \\sum^n_{i=1}{r_i} \\right]

    Where :math:`r_i` is the rank the model assigns to the positive edge,
    compared to the negative edges in the candidate list, and :math:`n` is the number of
    candidate lists, one per positive edge.

    AMRI values will be in the :math:`[-1, 1]` range, where 1 corresponds
    to optimal performance where each individual rank is 1. A value of 0 indicates
    model performance similar to a model assigning random scores, or equal score
    to every candidate. The value is negative if the model performs worse than the
    constant-score model."

    For more details see https://arxiv.org/abs/2002.06914

    Parameters
    ----------
    ranking : torch.Tensor
        ranking of each positive edge
    candidate_sizes : th.Tensor
        The size of each candidate list. If all candidate lists have
        the same size this will be a single-value tensor.

    Returns
    -------
    th.Tensor
        A single-value Tensor with the AMRI metric.

    .. versionadded: 0.4.0
    """
    if candidate_sizes.shape[0] > 1:
        assert ranking.shape[0] == candidate_sizes.shape[0], \
            ("ranking and candidate_sizes must have the same length, "
             f"got {ranking.shape=} {candidate_sizes.shape=}" )
        assert th.all(ranking <= candidate_sizes).item(), \
            "all ranks must be <= candidate_sizes"

    # We use the simplified form of AMRI calculation
    # 1 - \frac{MR-1}{E[MR-1]} = 1 - \frac{2*\sum_n{r-1}}{\sum_n{|S|}}
    # where n is the number of evaluations (number of positive edges),
    # r is the ranking of the positive edge in each ranked score list,
    # and |S| is the edge candidate set size.
    # See equation (8) in https://arxiv.org/abs/2002.06914
    nominator = 2 * th.sum(ranking - 1)
    if candidate_sizes.shape[0] == 1:
        denominator = candidate_sizes.item() * ranking.shape[0]
    else:
        denominator = th.sum(candidate_sizes)

    return 1 - th.div(nominator, denominator)
