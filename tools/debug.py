from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import RUNNERS




def main():
    cfg_file = 'projects/sniffyart/csra_1xb16_sniffyart-448px_hrnet.py'
    cfg = Config.fromfile(cfg_file)
    cfg.work_dir = './debug_workdir'

    runner = RUNNERS.build(cfg)
    runner.train()


if __name__ == '__main__':
    main()
