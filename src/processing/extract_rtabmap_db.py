

from typing import Dict

from src.experiment import Experiment
from src.processing.processor import Processor
from src.tools import check_file_path, run_command


class ExtractRtabmapDB(Processor):
    def __init__(self, rtabmap_binary:str="C:/Program Files/RTABMap/bin/rtabmap-export.exe"):
        super().__init__()
        self.rtabmap_binary = rtabmap_binary 

    def process(self,exp: Experiment):
        database = check_file_path(exp.get_experiment_folder(), "rtabmap.db")
        run_command(f'"{self.rtabmap_binary}" --poses --images --poses_format 1 {database}')
        for folder_name in [exp.rgb_path, exp.calib_path,exp.pose_path,exp.cloud_path]:
            check_file_path(folder_name)
