from SCDM_Driver import SCDM_DriverInterface
from model_loading import BasicModelLoader
import numpy
import json
import os
from pathlib import Path
from utils import LoggingColor

class ModelExecutor():
    """
    ModelExecutor
    ===============================================
    1. 執行模型
    2. 將需要的運算外交給 simulator 算
    3. 總結效能: 把 simulator 統整的資訊輸出到檔案
    
    ===============================================
    """

    LOG_DIR = Path(os.getcwd()).resolve() / "data" / "logs"
    DEFAULT_LOG_FILENAME = "ModelExecutorLog.json"
    
    _logger = LoggingColor.get_logger("executor")

    def __init__(self):
        self._current_loader: BasicModelLoader = None
        self._driver: SCDM_DriverInterface = None

    def config_driver(self, driver: SCDM_DriverInterface):
        """
        config all the things related to driver
        """
        self._driver = driver

    def load(self, model_loader: BasicModelLoader):
        """
        config model loader
        submit matrixes into driver
        """
        if self._driver is None:
            raise RuntimeError("Driver has not been configured. Please call config_driver() first.")
        
        # 如果已經有載入的 loader，先進行卸載以釋放資源
        if self._current_loader is not None:
            self.unload()
        self._current_loader = model_loader
        
        self._logger.info(f"Configuring loader: {model_loader.__class__.__name__}...")
        # 關鍵步驟：呼叫 Loader 的 config，這通常會觸發權重寫入 (driver.submit_matrix)
        self._current_loader.config(self._driver)

    def unload(self):
        """
        unload current model_loader if exists
        """
        if self._current_loader is not None:
            self._logger.info("Unloading current loader...")
            # 呼叫 Loader 的 unload，這通常會觸發清除矩陣 (driver.clear_matrix)
            self._current_loader.unload()
            self._current_loader = None

    def run(self, input_data: numpy.ndarray, input_command: str = None) -> numpy.ndarray:
        """
        pass input_data into model_loader
        user can invoke a loader-specified function via input_command
        """
        if self._current_loader is None:
            raise RuntimeError("No model loader is currently loaded. Please call load() first.")
        
        # 將執行工作轉發給 Loader，Loader 內部會決定哪些部分走 CPU，哪些呼叫 Driver
        return self._current_loader.run(input_data, input_command)

    def save_info(self, filename: str = None):
        """
        log statistic data in the driver into a file specified by path
        """
        if self._driver is None:
            self._logger.warning(LoggingColor.color_text("Driver not configured, cannot save stats.", LoggingColor.WARNING))
            return

        # 從 Driver 獲取統計資料
        stats = self._driver.get_statistic()
        
        if filename is None:
            filename = self.DEFAULT_LOG_FILENAME
        path = self.LOG_DIR / filename

        # 確保目錄存在
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # 將 Dict 寫入 JSON 檔案
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
            self._logger.info(LoggingColor.color_text(f"Statistics saved to {path}", LoggingColor.GREEN))
        except IOError as e:
            self._logger.error(LoggingColor.color_text(f"Error saving statistics: {e}", LoggingColor.ERROR))

    def clear_info_cache(self):
        """
        reset statistic data in the driver
        """
        if self._driver is not None:
            self._driver.reset_statistic()
            self._logger.info(LoggingColor.color_text("Driver statistics have been reset.", LoggingColor.GREEN))
