from SCDM_Driver import SCDM_DriverInterface
from model_loading import BasicModelLoader
import numpy


class ModelExecutor():
    """
    ModelExecutor
    ===============================================
    1. 執行模型
    2. 將需要的運算外交給 simulator 算
    3. 總結效能: 把 simulator 統整的資訊輸出到檔案
    
    ===============================================
    """
    def __init__(self):
        self._current_loader: BasicModelLoader = None
        pass

    def config_driver(self, driver: SCDM_DriverInterface):
        """
        config all the things related to driver
        """
        self._driver = driver
        pass

    def load(self, model_loader: BasicModelLoader):
        """
        config model loader
        submit matrixes into driver
        """
        self._current_loader = model_loader
        pass
    
    def unload(self):
        """
        unload current model_loader if exists
        """
        self._current_loader = None
        pass

    def run(self, input_data: numpy.ndarray, input_command: str = None) -> numpy.ndarray:
        """
        pass input_data into model_loader
        user can invoke a loader-specified function via input_command
        """
        pass

    def save_info(self, path: str):
        """
        log statistic data in the driver into a file specified by path
        """
        pass

    def clear_info_cache(self):
        """
        reset statistic data in the driver
        """
        pass
