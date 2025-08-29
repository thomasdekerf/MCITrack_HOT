from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.mcitrack.config import cfg, update_config_from_file


# Default model name whose checkpoint will be used if a custom
# experiment does not provide its own weights.  This avoids the
# need to train a new model for every hyper-parameter setting in the
# sweep.
BASE_YAML_NAME = "mcitrack_b224"


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file.  The sweep stores each
    # configuration in its own directory with a copy of the settings
    # file (config.yaml).  For backwards compatibility we first look
    # for experiments/mcitrack/<name>.yaml and, if that does not
    # exist, fall back to experiments/mcitrack/<name>/config.yaml.
    yaml_file = os.path.join(prj_dir, 'experiments/mcitrack/%s.yaml' % yaml_name)
    if not os.path.isfile(yaml_file):
        yaml_file = os.path.join(prj_dir, 'experiments/mcitrack', yaml_name, 'config.yaml')
    update_config_from_file(yaml_file)
    params.cfg = cfg

    params.yaml_name = yaml_name
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path.  When sweeping over parameters we may
    # generate a fresh yaml_name without a corresponding trained
    # checkpoint.  In that case, fall back to the checkpoint of the
    # base model defined by BASE_YAML_NAME.
    ckpt_dir = os.path.join(save_dir, 'checkpoints/train/mcitrack')
    ckpt_path = os.path.join(ckpt_dir, f"{yaml_name}/MCITRACK_ep{cfg.TEST.EPOCH:04d}.pth.tar")
    if not os.path.isfile(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, f"{BASE_YAML_NAME}/MCITRACK_ep{cfg.TEST.EPOCH:04d}.pth.tar")
    params.checkpoint = ckpt_path
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
