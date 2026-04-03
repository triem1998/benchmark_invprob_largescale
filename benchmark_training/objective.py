"""Training objective for unrolled models.

This objective is designed for both training and inference solvers.
It evaluates reconstruction quality using PSNR.
"""

import torch
from benchopt import BaseObjective
from deepinv.loss.metric import PSNR

from toolsbench.utils import save_comparison_figure


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
        self.psnr_metric = PSNR(max_pixel=max_pixel)
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.evaluation_count = 0
        self.ground_truth_shape = ground_truth_shape
        self.operator_norm = float(operator_norm)

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
        )

    def evaluate_result(self, name=None, reconstructions=None, val_psnr=None, **kwargs):
        """Compute the objective value given the output of a solver.

        For training-based solvers, this uses the pre-computed validation PSNR.
        For inference solvers, evaluates on the test set using reconstructions.

        Parameters
        ----------
        name : str, optional
            Name of the solver (provided by benchopt framework).
        reconstructions : list of torch.Tensor, optional
            List of reconstructed images.
        val_psnr : float, optional
            Pre-computed validation PSNR.
        train_psnr : float, optional
            Pre-computed training PSNR.
        **kwargs : dict
            Optional GPU and step metrics.

        Returns
        -------
        dict
            Dictionary with 'value' (negative PSNR for minimization) and metrics.
        """

        if val_psnr is not None:
            avg_psnr = val_psnr
        else:
            # compute from reconstructions
            if reconstructions is not None:
                with torch.no_grad():
                    local_psnr_sum = 0.0
                    local_count = 0
                    first_ground_truth = None
                    first_reconstruction = None

                    for batch_idx, (ground_truth, _) in enumerate(self.val_dataloader):
                        reconstruction = reconstructions[batch_idx].to(
                            ground_truth.device
                        )
                        # Save first image for visualization
                        if batch_idx == 0:
                            first_ground_truth = ground_truth
                            first_reconstruction = reconstruction

                        batch_psnr = self.psnr_metric(
                            reconstruction, ground_truth
                        ).item()
                        local_psnr_sum += batch_psnr * ground_truth.shape[0]
                        local_count += ground_truth.shape[0]

                    avg_psnr = local_psnr_sum / local_count if local_count > 0 else 0.0
            else:
                raise ValueError(
                    " must provide reconstructions or val_psnr for evaluation."
                )

        # Return value (primary metric for stopping criterion) and additional metrics
        result = dict(value=-avg_psnr, val_psnr=avg_psnr)

        self.evaluation_count += 1
        if reconstructions is not None and name is not None:
            output_dir = "evaluation_output/" + name.replace("/", "_").replace("..", "")
            save_comparison_figure(
                first_ground_truth,
                first_reconstruction,
                metrics={"psnr": avg_psnr},
                output_dir=output_dir,
                filename=f"eval_{self.evaluation_count:04d}.png",
                evaluation_count=self.evaluation_count,
            )

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
