"""Training objective for unrolled models.

This objective is designed for both training and inference solvers.
It evaluates reconstruction quality using PSNR.
"""

from benchopt import BaseObjective


class Objective(BaseObjective):
    """Training objective for unrolled reconstruction models.

    Evaluates reconstruction quality using PSNR.
    Supports both training solvers (with separate train/val sets) and
    inference solvers (with single dataset).
    """

    name = "reconstruction_objective"

    def set_data(
        self,
        train_dataloader=None,
        val_dataloader=None,
        physics=None,
        min_pixel=0.0,
        max_pixel=1.0,
        num_operators=None,
        ground_truth_shape=None,
        operator_norm=1.0,
        operator_norm_map=None,
    ):
        """Set the data from a Dataset to compute the objective.

        Parameters
        ----------
        train_dataloader : DataLoader, optional
            PyTorch DataLoader containing training ground truth and measurements.
        val_dataloader : DataLoader
            PyTorch DataLoader containing validation/test ground truth and measurements.
        physics : Physics
            Forward operator.
        min_pixel : float, optional
            Minimum pixel value for metrics. Default: 0.0.
        max_pixel : float, optional
            Maximum pixel value for metrics. Default: 1.0.
        num_operators : int, optional
            Number of operators in stacked physics.
        ground_truth_shape : tuple, optional
            Shape of ground truth.

        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.physics = physics
        self.num_operators = num_operators if num_operators is not None else 1
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.evaluation_count = 0
        self.ground_truth_shape = ground_truth_shape
        self.operator_norm = float(operator_norm)
        self.operator_norm_map = (
            operator_norm_map if operator_norm_map is not None else {}
        )

    def get_objective(self):
        """Returns a dict passed to Solver.set_objective method.

        Returns
        -------
        dict
            Dictionary with dataloader, physics, and metadata.
        """
        return dict(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            physics=self.physics,
            ground_truth_shape=self.ground_truth_shape,
            num_operators=self.num_operators,
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
            operator_norm=self.operator_norm,
            operator_norm_map=self.operator_norm_map,
        )

    def evaluate_result(self, val_psnr=None, **kwargs):
        """Compute the objective value given the output of a solver.

        Solvers always pre-compute and pass ``val_psnr`` directly.

        Parameters
        ----------
        val_psnr : float
            Pre-computed validation PSNR returned by the solver.
        **kwargs : dict
            Optional extra metrics (e.g. GPU memory, train PSNR).

        Returns
        -------
        dict
            Dictionary with 'value' (negative PSNR for minimization) and metrics.
        """
        if val_psnr is None:
            raise ValueError("Solver must provide val_psnr.")

        avg_psnr = val_psnr

        # Return value (primary metric for stopping criterion) and additional metrics
        result = dict(value=-avg_psnr, val_psnr=avg_psnr)

        self.evaluation_count += 1

        # Add all non-None metrics from kwargs to result
        for key, value in kwargs.items():
            if value is not None:
                result[key] = value

        return result

    def get_one_result(self):
        """Return one solution for which the objective can be evaluated.

        This is used by benchopt to check that the objective can be evaluated.

        Returns
        -------
        dict
            Dictionary with dummy result.
        """

        return {"name": "test_result", "val_psnr": 0.0}
