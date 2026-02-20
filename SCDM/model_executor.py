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
        # 如果已經有載入的 loader，先進行卸載以釋放資源
        if self._current_loader is not None:
            self.unload()
        self._current_loader = model_loader
        
        if self._driver is None:
            self._logger.warning(LoggingColor.color_text(
                "Driver has not been configured. Please call config_driver() first. " + 
                "The non-simulation version of TernaryBitNet will be used.", 
                LoggingColor.WARNING
            ))
        else:
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

        # 4. 收集執行資訊與統計數據
        driver_name = self._driver.__class__.__name__ if self._driver else "None"
        loader_name = self._current_loader.__class__.__name__
        
        # 紀錄輕量化的輸入維度資訊 (供 JSON 閱讀)
        input_info = {
            "type": type(input_data).__name__,
            "shape": list(input_data.shape) if hasattr(input_data, "shape") else None,
            "dtype": str(input_data.dtype) if hasattr(input_data, "dtype") else None
        }

        # 5. 將紀錄打包並存入歷史串列
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "loader": loader_name, 
            "loader_info": self._current_loader.loader_details(), 
            "driver": driver_name, 
            "input_command": input_command, 
            "input_data_info": input_info, 
            "duration_seconds": 0, 
            "driver_statistics": {}, 
            "_raw_input_data": input_data, 
            "_raw_output_data": result
        }

        if hasattr(result, "register_on_finish_callback") and callable(result.register_on_finish_callback):
            def on_finish_handler():
                # 這個 block 會在串流全部印完時才被執行
                end_time = time.time()
                record["duration_seconds"] = round(end_time - start_time, 4)
                if self._driver is not None:
                    # 抓取並清空剛才背景跑出來的硬體數據
                    record["driver_statistics"] = self._driver.get_statistic()
                    self._driver.reset_statistic()
                
                self._logger.info(LoggingColor.color_text(f"Stream finished in {record['duration_seconds']:.4f}s. Stats recorded.", LoggingColor.CYAN))

            # 註冊給 Tracker
            result.register_on_finish_callback(on_finish_handler)
            self._logger.info(LoggingColor.color_text("Stream generation started in background...", LoggingColor.CYAN))
        
        else:
            end_time = time.time()
            record["duration_seconds"] = round(end_time - start_time, 4)
            if self._driver is not None:
                record["driver_statistics"] = self._driver.get_statistic()
                self._driver.reset_statistic()
            self._logger.info(LoggingColor.color_text(f"Run completed in {record['duration_seconds']:.4f}s. Record saved internally.", LoggingColor.CYAN))
        
        self._history.append(record)
        return result

    def __save_vector(self, dir_path: Path, pickle_filename: str, raw_input_data) -> str:
        """ return pickle filename """
        pickle_filepath = dir_path / pickle_filename
        try:
            with open(pickle_filepath, 'wb') as pf:
                pickle.dump(raw_input_data, pf)
            return pickle_filename
        except Exception as e:
            self._logger.error(LoggingColor.color_text(f"Error saving pickle in {dir_path}: {e}", LoggingColor.ERROR))
            return None

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
            raw_input_data  = rec_copy.pop("_raw_input_data", None)
            raw_output_data = rec_copy.pop("_raw_output_data", None)
            if hasattr(raw_output_data, "get_full_result") and callable(raw_output_data.get_full_result):
                raw_output_data = raw_output_data.get_full_result()
            
            # 2. 儲存 Pickle
            rec_copy["input_pickle_file"]  = self.__save_vector(dir_path, "input.pickle",  raw_input_data)
            rec_copy["output_pickle_file"] = self.__save_vector(dir_path, "output.pickle", raw_output_data)

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
        reset statistic data in the driver & records in the executor
        """
        if self._driver is not None:
            self._driver.reset_statistic()
            self._logger.info(LoggingColor.color_text("Driver statistics have been reset.", LoggingColor.GREEN))
        self._history.clear()
        self._logger.info(LoggingColor.color_text("Executor records have been reset.", LoggingColor.GREEN))
