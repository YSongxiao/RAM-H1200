from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBE(nnUNetTrainer):
    """
    BE-specific nnU-Net trainer.

    Keeps the standard nnU-Net optimizer, loss, scheduler and architecture.
    Only adjusts training length and foreground oversampling for sparse BE lesions.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: str = "cuda",
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        self.num_epochs = 1000
        self.oversample_foreground_percent = 0.7
