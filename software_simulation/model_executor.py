from RRAM_Driver import RRAM_DriverInterface
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
    Model Executor
    ===========================================================================
    This class orchestrates the execution of machine learning models and handles 
    hardware integration and performance logging.

    Key Responsibilities:
    1. Execution Management: Executes the loaded model and tracks its runtime.
    2. Hardware Offloading: Delegates matrix computations to the configured 
       RRAM driver or simulator via the Model Loader.
    3. Performance Summarization: Collects execution statistics and exports 
       logs and binary data to the designated local storage.
    
    ===========================================================================
    """

    LOG_DIR = Path(os.getcwd()).resolve() / "data" / "logs"
    DEFAULT_LOG_PREFIX = "ME"
    
    _logger = LoggingColor.get_logger("executor")

    def __init__(self):
        self._current_loader: BasicModelLoader = None
        self._driver: RRAM_DriverInterface = None
        self._history = list()

    def config_driver(self, driver: RRAM_DriverInterface):
        """
        Configure the hardware driver for the executor.

        Args:
            driver (RRAM_DriverInterface): The initialized hardware driver.
        """
        self._driver = driver

    def load(self, model_loader: BasicModelLoader):
        """
        Mount a model loader and trigger hardware configuration.

        If another loader is currently active, it unloads it first to release 
        resources. Then, it submits the necessary matrices to the hardware driver.

        Args:
            model_loader (BasicModelLoader): The model loader to be executed.
        """
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
            self._current_loader.config(self._driver)

    def unload(self):
        """
        Unload the current model loader and release hardware resources.
        """
        if self._current_loader is not None:
            self._logger.info("Unloading current loader...")
            self._current_loader.unload()
            self._current_loader = None

    def run(self, input_data: numpy.ndarray, input_command: str = None) -> numpy.ndarray:
        """
        Pass input data into the model loader and track execution statistics.

        Handles both standard synchronous execution and asynchronous stream generation.

        Args:
            input_data (numpy.ndarray): The processed input data/tokens.
            input_command (str, optional): The command specifying the execution mode.

        Returns:
            numpy.ndarray: The result data or stream tracker from the model loader.

        Raises:
            RuntimeError: If no model loader is currently loaded.
        """
        if self._current_loader is None:
            raise RuntimeError("No model loader is currently loaded. Please call load() first.")
        
        start_time = time.time()

        result = self._current_loader.run(input_data, input_command)

        driver_name = self._driver.__class__.__name__ if self._driver else "None"
        loader_name = self._current_loader.__class__.__name__
        
        input_info = {
            "type": type(input_data).__name__,
            "shape": list(input_data.shape) if hasattr(input_data, "shape") else None,
            "dtype": str(input_data.dtype) if hasattr(input_data, "dtype") else None
        }

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
                end_time = time.time()
                record["duration_seconds"] = round(end_time - start_time, 4)
                if self._driver is not None:
                    record["driver_statistics"] = self._driver.get_statistic()
                    self._driver.reset_statistic()
                
                self._logger.info(LoggingColor.color_text(f"Stream finished in {record['duration_seconds']:.4f}s. Stats recorded.", LoggingColor.CYAN))

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
        """
        Helper method to save raw data as a pickle file.
        
        Returns:
            str: The filename if successful, None otherwise.
        """
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
        Export logged statistics and raw data to the local file system.

        Creates a unique directory for each execution record containing a JSON 
        log file and pickle files for the raw inputs and outputs.

        Args:
            prefix (str, optional): The prefix for the log directory name. 
                                    Defaults to DEFAULT_LOG_PREFIX.
        """
        if not self._history:
            self._logger.warning(LoggingColor.color_text("No history records to save.", LoggingColor.WARNING))
            return

        if prefix is None:
            prefix = self.DEFAULT_LOG_PREFIX
        
        saved_count = 0
        for record in self._history:
            safe_time = record["timestamp"].replace("-", "").replace(":", "").replace(" ", "_")
            run_folder_name = f"{prefix}_{safe_time}_{record['loader']}_{record['driver']}"
            dir_path = self.LOG_DIR / run_folder_name
            os.makedirs(dir_path, exist_ok=True)

            rec_copy = record.copy()
            
            raw_input_data  = rec_copy.pop("_raw_input_data", None)
            raw_output_data = rec_copy.pop("_raw_output_data", None)
            
            if hasattr(raw_output_data, "get_full_result") and callable(raw_output_data.get_full_result):
                raw_output_data = raw_output_data.get_full_result()
            
            rec_copy["input_pickle_file"]  = self.__save_vector(dir_path, "input.pickle",  raw_input_data)
            rec_copy["output_pickle_file"] = self.__save_vector(dir_path, "output.pickle", raw_output_data)

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
        Clear the internal execution history and reset hardware statistics.
        """
        if self._driver is not None:
            self._driver.reset_statistic()
            self._logger.info(LoggingColor.color_text("Driver statistics have been reset.", LoggingColor.GREEN))
        self._history.clear()
        self._logger.info(LoggingColor.color_text("Executor records have been reset.", LoggingColor.GREEN))
