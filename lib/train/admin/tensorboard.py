import os
from collections import OrderedDict
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('WARNING: You are using tensorboardX instead sis you have a too old pytorch version.')
    from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self, directory, loader_names):
        self.directory = directory
        self.writer = OrderedDict({name: SummaryWriter(os.path.join(self.directory, name)) for name in loader_names})

    def write_info(self, script_name, description):
        tb_info_writer = SummaryWriter(os.path.join(self.directory, 'info'))
        tb_info_writer.add_text('Script_name', script_name)
        tb_info_writer.add_text('Description', description)
        tb_info_writer.close()

    def write_epoch(self, stats: OrderedDict, epoch: int, ind=-1):
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for var_name, val in loader_stats.items():
                if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                    self.writer[loader_name].add_scalar(var_name, val.history[ind], epoch)
            # Flush to ensure TensorBoard reads the latest values
            self.writer[loader_name].flush()

    def write_interval(self, loader_name: str, stats: OrderedDict, step: int):
        """Write current average stats to TensorBoard at a given step.

        Args:
            loader_name: Name of the dataloader whose stats should be written.
            stats: OrderedDict mapping loader names to their statistics.
            step: Global step (e.g. iteration number) for TensorBoard.
        """
        loader_stats = stats.get(loader_name, None)
        if loader_stats is None:
            return

        for var_name, val in loader_stats.items():
            if hasattr(val, 'avg'):
                self.writer[loader_name].add_scalar(var_name, val.avg, step)

        # Flush to ensure TensorBoard reads the latest values
        self.writer[loader_name].flush()
