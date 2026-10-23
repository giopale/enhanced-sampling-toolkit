import hydra
import importlib
from rich import print


from logger_setup import logger



@hydra.main(config_path="config", config_name="common", version_base=None)
def main(cfg):
    module_path = cfg._target_.rsplit(".", 1) # get module path e.g. command.reweight-property
    module = importlib.import_module(module_path[0]) # import module
    func = getattr(module, module_path[1]) # get right function from module e.g. run
    
    print("[bold magenta]Executing command:[/bold magenta]", cfg._target_)
    func(cfg)

if __name__ == "__main__":
    main()

