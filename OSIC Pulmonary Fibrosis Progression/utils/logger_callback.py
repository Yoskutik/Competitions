from tensorflow.keras.callbacks import Callback
from awesome_progress_bar import ProgressBar


class LoggerCallback(Callback):
    def __init__(self, total):
        self._total = total
        
    def on_epoch_begin(self, epoch, logs):
        print(f'Epoch {epoch}:')
        prefix = self._create_prefix(0).rstrip()
        self.bar = ProgressBar(self._total, prefix, '', bar_length=80, new_line_at_end=False)
    
    def on_train_batch_end(self, batch, logs):
        self.bar.prefix = self._create_prefix(batch)
        self.bar.iter(f'   mape: {logs["mape"]:>6.2f}')
    
    def on_test_end(self, logs):
        print(f' - val_mape: {logs["mape"]:>6.2f}')
        
    def _create_prefix(self, batch):
        prefix = f'{batch}/{self._total - 1} '
        return prefix.rjust(len(str(self._total)) * 2 + 4)