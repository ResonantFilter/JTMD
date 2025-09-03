from typing import Union
import wandb
import torch
import uuid
import sys
import signal
import sys
import argparse
# wandb.require("core")

def debugger_is_active() -> bool:
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def graceful_stop(sig, frame):
    signal.signal(sig, signal.SIG_IGN)
    print("Gracefully stopping wandb...")
    wandb.finish()
    sys.exit(0)
    
def slurm_wandb_argparser():
    parser = argparse.ArgumentParser(add_help=False)
    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default="Vast&Spurious-W")
    parser.add_argument("--wandb_entity", type=str, default="MLV_Kitchen")

    # SLURM
    parser.add_argument("--slurm_job_name", type=str)
    parser.add_argument("--slurm_constraint", type=str)
    parser.add_argument("--slurm_partition", type=str)
    parser.add_argument("--slurm_mem_gb", type=int, default=128)
    parser.add_argument("--slurm_log_dir", type=str, default="exp/logs")

    return parser

# class WandbWrapper:
#     def __init__(self, project_name: str, config: Union[dict, None] = None) -> None:
#         self.project_name:  str                 = project_name
#         self.run_name:      Union[str, None]    = str(uuid.uuid1())
#         self.config:        Union[dict, None]   = config

#         if debugger_is_active():
#             print("Debugging, logging to W&B disabled")
#             return None
#         else:
#             self.initialize()
#             signal.signal(signal.SIGINT, graceful_stop)
#             self.backend = wandb


#     def initialize(self):
#         wandb.init(
#             project=self.project_name, 
#             name=self.run_name, 
#             config=self.config
#         )

#     def log_model(self, model: torch.nn.Module, model_name: str = "model"):
#         torch.save(model.state_dict(), f"{model_name}.pth")
#         wandb.save(f"{model_name}.pth")
#         wandb.log_artifact(f"{model_name}.pth", name=model_name, type="model")

#     def log_output(self, output, step=None):
#         try:
#             wandb.log(output, step=step)
#         except Exception as e:
#             print("An issue occured during wb logging", e.__traceback__)
#             pass
        
#     def log(self, dict, step=None):
#         return wandb.log(dict, step=step)

#     def finish(self):
#         wandb.finish()

