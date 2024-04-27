from pytorch_lightning.callbacks import ModelCheckpoint
class SaveCheckpoint(ModelCheckpoint):
    """save checkpoint after each training epoch without validation.
    if ``last_k == -1``, all models are saved. and no monitor needed in this condition.
    otherwise, please log ``global_step`` in the training_step. e.g. self.log('global_step', self.global_step)

    :param last_k: the latest k models will be saved.
    :param save_weights_only: if ``True``, only the model's weights will be saved,
    else the full model is saved.
    """
    def __init__(self, last_k=5, save_weights_only=False):
        if last_k == -1:
            super().__init__(save_top_k=-1, save_last=False, save_weights_only=save_weights_only)
        else:
            super().__init__(monitor='global_step', mode='max', save_top_k=last_k,
                             save_last=False, save_weights_only=save_weights_only)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """
        save checkpoint after each train epoch
        """
        self.save_checkpoint(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        """
        overwrite the methods in ModelCheckpoint to avoid save checkpoint on the end of the val loop
        """
        pass