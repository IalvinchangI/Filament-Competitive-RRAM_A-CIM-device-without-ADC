from SCDM_Driver import SCDM_DriverInterface
from model_loading import BasicModelLoader
import numpy
import json
import os
import time
import pickle
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
    DEFAULT_LOG_PREFIX = "ME"
    
    _logger = LoggingColor.get_logger("executor")

    def __init__(self):
        self._current_loader: BasicModelLoader = None
        self._driver: SCDM_DriverInterface = None
        self._history = list()

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
        
        # 1. 紀錄開始時間
        start_time = time.time()

        # 2. 轉發給 Loader 執行
        result = self._current_loader.run(input_data, input_command)

        # 3. 紀錄結束時間並計算花費時間
        end_time = time.time()
        duration = end_time - start_time

        # 4. 收集執行資訊與統計數據
        driver_name = self._driver.__class__.__name__ if self._driver else "None"
        loader_name = self._current_loader.__class__.__name__
        
        # 紀錄輕量化的輸入維度資訊 (供 JSON 閱讀)
        input_info = {
            "type": type(input_data).__name__,
            "shape": list(input_data.shape) if hasattr(input_data, "shape") else None,
            "dtype": str(input_data.dtype) if hasattr(input_data, "dtype") else None
        }

        run_stats = {}
        if self._driver is not None:
            # 取得該次 run 期間累積的數據
            run_stats = self._driver.get_statistic()
            # 刷新歸零，準備下一次 run
            self._driver.reset_statistic()

        # 5. 將紀錄打包並存入歷史串列
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "loader": loader_name, 
            "driver": driver_name, 
            "input_command": input_command, 
            "input_data_info": input_info, 
            "duration_seconds": round(duration, 4), 
            "driver_statistics": run_stats, 
            "_raw_input_data": input_data
        }
        
        self._history.append(record)
        self._logger.info(LoggingColor.color_text(f"Run completed in {duration:.4f}s. Record saved internally.", LoggingColor.CYAN))

        return result

    def save_info(self, prefix: str = None):
        """
        log statistic data in the driver into a file specified by path
        """
        if not self._history:
            self._logger.warning(LoggingColor.color_text("No history records to save.", LoggingColor.WARNING))
            return

        if prefix is None:
            prefix = self.DEFAULT_LOG_PREFIX
        
        saved_count = 0
        for record in self._history:
            # 1. 建立該次執行的專屬資料夾
            # 將時間格式轉換為安全的資料夾名稱 (ex: 2026-02-19 14:30:00 -> 20260219_143000)
            safe_time = record["timestamp"].replace("-", "").replace(":", "").replace(" ", "_")
            run_folder_name = f"{prefix}_{safe_time}_{record['loader']}_{record['driver']}"
            dir_path = self.LOG_DIR / run_folder_name
            os.makedirs(dir_path, exist_ok=True)

            # 淺拷貝一份，以免修改到還在記憶體中的原始歷史紀錄
            rec_copy = record.copy()
            
            # 抽出 raw data
            raw_input_data = rec_copy.pop("_raw_input_data", None)
            
            # 2. 儲存 Pickle (只存 input_data)
            pickle_filepath = dir_path / "input.pickle"
            try:
                with open(pickle_filepath, 'wb') as pf:
                    pickle.dump(raw_input_data, pf)
                rec_copy["input_pickle_file"] = "input.pickle"
            except Exception as e:
                self._logger.error(LoggingColor.color_text(f"Error saving pickle in {run_folder_name}: {e}", LoggingColor.ERROR))
                rec_copy["input_pickle_file"] = "ERROR_SAVING_PICKLE"

            # 3. 儲存 JSON (包含此次執行所有的 log 與統計)
            json_filepath = dir_path / "log.json"
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(rec_copy, f, indent=4, ensure_ascii=False)
                saved_count += 1
            except IOError as e:
                self._logger.error(LoggingColor.color_text(f"Error saving JSON in {run_folder_name}: {e}", LoggingColor.ERROR))

        self._logger.info(LoggingColor.color_text(f"Successfully saved {saved_count} execution records to {self.LOG_DIR}", LoggingColor.GREEN))

    def clear_info_cache(self):
        """
        reset statistic data in the driver
        """
        if self._driver is not None:
            self._driver.reset_statistic()
            self._logger.info(LoggingColor.color_text("Driver statistics have been reset.", LoggingColor.GREEN))
