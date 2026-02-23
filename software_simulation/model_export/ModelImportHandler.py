from typing_extensions import Callable, Union
import logging
from utils import LoggingColor
from utils.environment_variables import TORCH_DEVICE
import torch
from pathlib import Path
import os

class ModelImportHandler():
    """
    Model Import Handler
    ===========================================================================
    A handler class to manage the registration, batch importation, and local storage 
    of PyTorch model loading functions.

    Key Responsibilities:
    1. Central Registry: Acts as a centralized dictionary mapping model names to 
       their respective initialization/download functions.
    2. Caching Mechanism: Ensures models are only downloaded or processed if they 
       are not already available locally, preventing redundant operations.
    3. Storage Management: Automatically saves imported models to a designated 
       local directory.

    ===========================================================================
    """

    MODEL_DEFAULT_DIR = Path(os.getcwd()).resolve() / "data" / "model"
    MODEL_FILE_EXTENTION = ".pickle" 

    DEVICE = TORCH_DEVICE

    __model_list: dict[str, Callable] = dict()
    __logger = LoggingColor.get_logger("ModelImportHandler")

    @classmethod
    def get_model_list(cls) -> dict:
        """
        Retrieve a copy of the currently registered model list.
        
        Returns:
            dict[str, Callable]: A copy of the dictionary containing model names 
                                 and their import functions.
        """
        return cls.__model_list.copy()

    @classmethod
    def register_model(cls, model_name: str):
        """
        A decorator factory to register a function as a model importer.
        
        Args:
            model_name (str): The unique identifier/name for the model being registered.

        Returns:
            function: The decorator function that registers the importer.
        """
        def decorator(import_function: Callable):
            if model_name in cls.__model_list:
                cls.__logger.warning(LoggingColor.color_text(f"{model_name} Import Twice", LoggingColor.WARNING))
            
            cls.__model_list[model_name] = import_function
            return import_function
        return decorator

    @classmethod
    def import_all_model(cls, force_update: bool = False):
        """
        Iterate through all registered functions to download or cache models locally.

        Checks if the model file already exists. If it exists, skips the import 
        (unless force_update is True). If not, executes the import function and saves the result.

        Args:
            force_update (bool, optional): If True, re-imports and overwrites existing models. 
                                           Defaults to False.
        """
        if not cls.MODEL_DEFAULT_DIR.exists():
             try:
                cls.MODEL_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
             except Exception as e:
                cls.__logger.error(LoggingColor.color_text(f"Failed to create directory {cls.MODEL_DEFAULT_DIR} | {e}", LoggingColor.ERROR))
                return

        for model_name, import_function in cls.__model_list.items():
            try:
                file_path = cls.MODEL_DEFAULT_DIR / (model_name + cls.MODEL_FILE_EXTENTION)

                if file_path.exists():
                    if force_update:
                        cls.__logger.info(f"Force update enabled. Re-importing {model_name}...")
                    else:
                        cls.__logger.info(f"Model {model_name} found at {file_path}. {LoggingColor.color_text('Skipping download/import.', LoggingColor.GREEN)}")
                        continue
                
                model_instance = import_function()
                
                if model_instance is not None:
                    cls.__logger.info(f"Successfully Imported {model_name}")
                    cls.store_model(model_instance, model_name)
                else:
                    cls.__logger.warning(LoggingColor.color_text(f"{model_name} returned None, skipping storage.", LoggingColor.WARNING))

            except Exception as e:
                cls.__logger.error(LoggingColor.color_text(f"Import {model_name} Failed | {e}", LoggingColor.ERROR))
    
    @classmethod
    def store_model(cls, model, model_name: str, path: Union[str, None] = None):
        """
        Save the provided PyTorch model object to the specified directory.

        Args:
            model (torch.nn.Module): The PyTorch model instance to save.
            model_name (str): The name of the model (used for the filename).
            path (Union[str, None], optional): Custom directory path. Defaults to MODEL_DEFAULT_DIR.
        """
        if path is None:
            save_dir = cls.MODEL_DEFAULT_DIR
        else:
            save_dir = Path(path)
        
        if not save_dir.exists():
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                cls.__logger.debug(f"Created directory: {save_dir}")
            except Exception as e:
                cls.__logger.error(LoggingColor.color_text(f"Failed to create directory {save_dir} | {e}", LoggingColor.ERROR))
                return

        if isinstance(model, torch.nn.Module):
            try:
                file_path = save_dir / (model_name + cls.MODEL_FILE_EXTENTION)
                torch.save(model, file_path)
                cls.__logger.info(LoggingColor.color_text(f"Stored {model_name} at {file_path}", LoggingColor.GREEN))
            except Exception as e:
                cls.__logger.error(LoggingColor.color_text(f"Failed to save {model_name} to disk | {e}", LoggingColor.ERROR))
        else:
            cls.__logger.error(LoggingColor.color_text(f"Skipping storage for {model_name}: Expected torch.nn.Module, got {type(model).__name__}", LoggingColor.ERROR))
        

    @classmethod
    def silence(cls, tf: bool):
        """
        Control the logging verbosity of the handler.

        Args:
            tf (bool): If True, sets the logging level to ERROR (suppresses warnings/info). 
                       If False, sets the logging level to DEBUG (default).
        """
        if tf == True:
            cls.__logger.setLevel(level=logging.ERROR)
        else:
            cls.__logger.setLevel(level=logging.DEBUG)
