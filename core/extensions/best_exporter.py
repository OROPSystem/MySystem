from __future__ import absolute_import

import glob
import json
import os
import shutil

from tensorflow.python.estimator import gc
from tensorflow.python.estimator import util
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.exporter import Exporter, _SavedModelExporter
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary_iterator


def _verify_compare_fn_args(compare_fn):
    """Verifies compare_fn arguments."""
    args = set(util.fn_args(compare_fn))
    if "best_eval_result" not in args:
        raise ValueError(
            "compare_fn (%s) must include best_eval_result argument." % compare_fn
        )
    if "current_eval_result" not in args:
        raise ValueError(
            "compare_fn (%s) must include current_eval_result argument." % compare_fn
        )
    non_valid_args = list(args - {"best_eval_result", "current_eval_result"})
    if non_valid_args:
        raise ValueError(
            "compare_fn (%s) has following not expected args: %s"
            % (compare_fn, non_valid_args)
        )


def _loss_smaller(best_eval_result, current_eval_result):
    """Compares two evaluation results and returns true if the 2nd one is smaller.
    Both evaluation results should have the values for MetricKeys.LOSS, which are
    used for comparison.
    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.
    Returns:
      True if the loss of current_eval_result is smaller; otherwise, False.
    Raises:
      ValueError: If input eval result is None or no loss is available.
    """
    default_key = metric_keys.MetricKeys.LOSS
    if not best_eval_result or default_key not in best_eval_result:
        raise ValueError("best_eval_result cannot be empty or no loss is found in it.")

    if not current_eval_result or default_key not in current_eval_result:
        raise ValueError(
            "current_eval_result cannot be empty or no loss is found in it."
        )

    return best_eval_result[default_key] > current_eval_result[default_key]


class BestExporter(Exporter):
    """This class exports the serving graph and checkpoints of the best models.
    This class performs a model export everytime when the new model is better
    than any exsiting model.
    """

    def __init__(
        self,
        name="best_exporter",
        serving_input_receiver_fn=None,
        event_file_pattern="eval/*.tfevents.*",
        compare_fn=_loss_smaller,
        assets_extra=None,
        as_text=False,
        exports_to_keep=5,
    ):
        """Create an `Exporter` to use with `tf.estimator.EvalSpec`.
        Example of creating a BestExporter for training and evluation:
        ```python
        def make_train_and_eval_fn():
          # Set up feature columns.
          categorial_feature_a = (
              tf.feature_column.categorical_column_with_hash_bucket(...))
          categorial_feature_a_emb = embedding_column(
              categorical_column=categorial_feature_a, ...)
          ...  # other feature columns
          estimator = tf.estimator.DNNClassifier(
              config=tf.estimator.RunConfig(
                  model_dir='/my_model', save_summary_steps=100),
              feature_columns=[categorial_feature_a_emb, ...],
              hidden_units=[1024, 512, 256])
          serving_feature_spec = tf.feature_column.make_parse_example_spec(
              categorial_feature_a_emb)
          serving_input_receiver_fn = (
              tf.estimator.export.build_parsing_serving_input_receiver_fn(
              serving_feature_spec))
          exporter = tf.estimator.BestExporter(
              name="best_exporter",
              serving_input_receiver_fn=serving_input_receiver_fn,
              exports_to_keep=5)
          train_spec = tf.estimator.TrainSpec(...)
          eval_spec = [tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=100,
            exporters=exporter,
            start_delay_secs=0,
            throttle_secs=5)]
          return tf.estimator.DistributedTrainingSpec(estimator, train_spec,
                                                      eval_spec)
        ```
        Args:
          name: unique name of this `Exporter` that is going to be used in the
            export path.
          serving_input_receiver_fn: a function that takes no arguments and returns
            a `ServingInputReceiver`.
          event_file_pattern: event file name pattern relative to model_dir. If
            None, however, the exporter would not be preemption-safe. To bex
            preemption-safe, event_file_pattern should be specified.
          compare_fn: a function that compares two evaluation results and returns
            true if current evaluation result is better. Follows the signature:
            * Args:
              * `best_eval_result`: This is the evaluation result of the best model.
              * `current_eval_result`: This is the evaluation result of current
                     candidate model.
            * Returns:
              True if current evaluation result is better; otherwise, False.
          assets_extra: An optional dict specifying how to populate the assets.extra
            directory within the exported SavedModel.  Each key should give the
            destination path (including the filename) relative to the assets.extra
            directory.  The corresponding value gives the full path of the source
            file to be copied.  For example, the simple case of copying a single
            file without renaming it is specified as `{'my_asset_file.txt':
            '/path/to/my_asset_file.txt'}`.
          as_text: whether to write the SavedModel proto in text format. Defaults to
            `False`.
          exports_to_keep: Number of exports to keep.  Older exports will be
            garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
            collection.
        Raises:
          ValueError: if any arguments is invalid.
        """
        self._compare_fn = compare_fn
        if self._compare_fn is None:
            raise ValueError("`compare_fn` must not be None.")
        _verify_compare_fn_args(self._compare_fn)

        self._saved_model_exporter = _SavedModelExporter(
            name, serving_input_receiver_fn, assets_extra, as_text
        )

        self._event_file_pattern = event_file_pattern
        self._model_dir = None
        self._best_eval_result = None

        self._exports_to_keep = exports_to_keep
        self._log = {}
        if exports_to_keep is not None and exports_to_keep <= 0:
            raise ValueError("`exports_to_keep`, if provided, must be positive number")

    @property
    def name(self):
        return self._saved_model_exporter.name

    def export(
        self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export
    ):
        export_result = None

        if self._model_dir != estimator.model_dir and self._event_file_pattern:
            # Loads best metric from event files.
            tf_logging.info("Loading best metric from event files.")

            self._model_dir = estimator.model_dir
            full_event_file_pattern = os.path.join(
                self._model_dir, self._event_file_pattern
            )
            self._best_eval_result = self._get_best_eval_result(full_event_file_pattern)
        if os.path.isfile(os.path.join(export_path, "export.log")):
            self._log = {}
            try:
                self._log = json.load(
                    open(os.path.join(export_path, "export.log"), "r")
                )
            except json.JSONDecodeError:
                pass
            if len(self._log) == 0:
                self._best_eval_result = None

        if self._best_eval_result is None or self._compare_fn(
            best_eval_result=self._best_eval_result, current_eval_result=eval_result
        ):
            tf_logging.info("Performing best model export.")
            self._best_eval_result = eval_result

            export_result = self._saved_model_exporter.export(
                estimator,
                export_path,
                checkpoint_path,
                eval_result,
                is_the_final_export,
            )
            export_result_path = export_result.decode("utf-8")
            self._log[export_result_path] = {
                k: float(v) for k, v in eval_result.items()
            }
            self._copy_checkpoint(
                checkpoint_path, export_result_path, eval_result["global_step"]
            )

            self._garbage_collect_exports(export_path)
            with open(os.path.join(export_path, "export.log"), "w") as fp:
                json.dump(self._log, fp)

        return export_result

    def _copy_checkpoint(self, checkpoint_pattern, dest_path, step):
        for file in glob.glob(checkpoint_pattern + "*"):
            shutil.copy(file, dest_path)
        with open(os.path.join(dest_path, "checkpoint"), "w") as fp:
            text = 'model_checkpoint_path: "model.ckpt-number"\n'.replace(
                "number", str(step)
            )
            fp.write(text)
            fp.close()

    def _garbage_collect_exports(self, export_dir_base):
        """Deletes older exports, retaining only a given number of the most recent.
        Export subdirectories are assumed to be named with monotonically increasing
        integers; the most recent are taken to be those with the largest values.
        Args:
          export_dir_base: the base directory under which each export is in a
            versioned subdirectory.
        """
        if self._exports_to_keep is None:
            return

        def _export_version_parser(path):
            # create a simple parser that pulls the export_version from the directory.
            filename = os.path.basename(path.path)
            if not (len(filename) == 10 and filename.isdigit()):
                return None
            return path._replace(export_version=int(filename))

        # pylint: disable=protected-access
        keep_filter = gc._largest_export_versions(self._exports_to_keep)
        delete_filter = gc._negation(keep_filter)

        for p in delete_filter(
            gc._get_paths(export_dir_base, parser=_export_version_parser)
        ):
            try:
                del self._log[p.path]
                gfile.DeleteRecursively(p.path)
            except errors_impl.NotFoundError as e:
                tf_logging.warn("Can not delete %s recursively: %s", p.path, e)
        # pylint: enable=protected-access

    def _get_best_eval_result(self, event_files):
        """Get the best eval result from event files.
        Args:
          event_files: Absolute pattern of event files.
        Returns:
          The best eval result.
        """
        if not event_files:
            return None
        event_count = 0
        best_eval_result = None
        for event_file in gfile.Glob(os.path.join(event_files)):
            for event in summary_iterator.summary_iterator(event_file):
                if event.HasField("summary"):
                    event_eval_result = {}
                    for value in event.summary.value:
                        if value.HasField("simple_value"):
                            event_eval_result[value.tag] = value.simple_value
                    if event_eval_result:
                        if best_eval_result is None or self._compare_fn(
                            best_eval_result, event_eval_result
                        ):
                            event_count += 1
                            best_eval_result = event_eval_result
        if event_count < 2:
            return None
        return best_eval_result
