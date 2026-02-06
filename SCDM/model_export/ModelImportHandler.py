from typing_extensions import Callable, Union
import logging
from utils import LoggingColor
import torch
from pathlib import Path
import os

class ModelImportHandler():
    """
    A handler class to manage the registration, batch importation, and storage of model loading functions.
    It acts as a central registry and ensures models are only downloaded/processed if not already locally available.
    """

    # Define default storage path and extension
    MODEL_DEFAULT_DIR = Path(os.getcwd()).resolve() / "data" / "model"
    MODEL_FILE_EXTENTION = ".pickle" # torch.save uses pickle by default

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # A private dictionary to store the mapping between model names and their import functions.
    __model_list: dict[str, Callable] = {}

    @classmethod
    def get_model_list(cls):
        """
        Retrieves a copy of the currently registered model list.
        
        Returns:
            dict[str, Callable]: A copy of the dictionary containing model names and import functions.
        """
        return cls.__model_list.copy()

    # Initialize the logger for this class.
    __logger = LoggingColor.get_logger("ModelImportHandler")

    @classmethod
    def register_model(cls, model_name: str):
        """
        A decorator factory to register a function as a model importer.
        
        Usage:
            @ModelImportHandler.register_model("MyModelName")
            def import_my_model():
                # ... load model ...
                return model_instance

        Args:
            model_name (str): The unique identifier/name for the model being registered.

        Returns:
            function: The decorator function that registers the importer.
        """
        def decorator(import_function: Callable):
            # Check if the model name is already registered to warn about duplicates.
            if model_name in cls.__model_list:
                cls.__logger.warning(LoggingColor.color_text(f"{model_name} Import Twice", LoggingColor.WARNING))
            
            # Register the function in the dictionary.
            cls.__model_list[model_name] = import_function
            return import_function
        return decorator

    @classmethod
    def import_all_model(cls, force_update: bool = False):
        """
        Iterates through all registered functions.
        Checks if the model file already exists locally.
        If it exists, skips the import (unless force_update is True).
        If not, executes the import function and saves the result.

        Args:
            force_update (bool): If True, re-imports and overwrites existing models.
                                 Defaults to False.
        """
        # Ensure the directory exists before checking files
        if not cls.MODEL_DEFAULT_DIR.exists():
             try:
                cls.MODEL_DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
             except Exception as e:
                cls.__logger.error(LoggingColor.color_text(f"Failed to create directory {cls.MODEL_DEFAULT_DIR} | {e}", LoggingColor.ERROR))
                return

        for model_name, import_function in cls.__model_list.items():
            try:
                # Construct the expected file path
                file_path = cls.MODEL_DEFAULT_DIR / (model_name + cls.MODEL_FILE_EXTENTION)

                # Cache Check
                if file_path.exists():
                    if force_update:
                        cls.__logger.info(f"Force update enabled. Re-importing {model_name}...")
                    else:
                        cls.__logger.info(f"Model {model_name} found at {file_path}. {LoggingColor.color_text('Skipping download/import.', LoggingColor.GREEN)}")
                        continue
                
                # Execute the registered import function (Download/Load logic)
                model_instance = import_function()
                
                # Automatically store the imported model to disk
                if model_instance is not None:
                    cls.__logger.info(f"Successfully Imported {model_name}")
                    cls.store_model(model_instance, model_name)
                else:
                    cls.__logger.warning(LoggingColor.color_text(f"{model_name} returned None, skipping storage.", LoggingColor.WARNING))

            except Exception as e:
                # Log any errors encountered during the import execution.
                cls.__logger.error(LoggingColor.color_text(f"Import {model_name} Failed | {e}", LoggingColor.ERROR))
    
    @classmethod
    def store_model(cls, model, model_name, path: Union[str, None] = None):
        """
        Saves the provided PyTorch model object to the specified directory using torch.save.

        Args:
            model (torch.nn.Module): The model instance to save.
            model_name (str): The name of the model (used for the filename).
            path (Union[str, None], optional): Custom directory path. Defaults to MODEL_DEFAULT_DIR.
        """
        # Determine the directory path
        if path is None:
            save_dir = cls.MODEL_DEFAULT_DIR
        else:
            save_dir = Path(path)
        
        # Ensure the directory exists
        if not save_dir.exists():
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                cls.__logger.debug(f"Created directory: {save_dir}")
            except Exception as e:
                cls.__logger.error(LoggingColor.color_text(f"Failed to create directory {save_dir} | {e}", LoggingColor.ERROR))
                return

        # Check if it is a valid PyTorch model
        if isinstance(model, torch.nn.Module):
            try:
                # Construct full file path
                file_path = save_dir / (model_name + cls.MODEL_FILE_EXTENTION)
                
                # Save the entire model object
                torch.save(model, file_path)
                
                # Log success
                cls.__logger.info(LoggingColor.color_text(f"Stored {model_name} at {file_path}", LoggingColor.GREEN))
            except Exception as e:
                # Log save error
                cls.__logger.error(LoggingColor.color_text(f"Failed to save {model_name} to disk | {e}", LoggingColor.ERROR))
        else:
            # Log type error (Not a PyTorch Module)
            cls.__logger.error(LoggingColor.color_text(f"Skipping storage for {model_name}: Expected torch.nn.Module, got {type(model).__name__}", LoggingColor.ERROR))
        

    @classmethod
    def silence(cls, tf: bool):
        """
        Controls the logging verbosity of the handler.

        Args:
            tf (bool): If True, sets logging level to ERROR (suppresses warnings/info).
                       If False, sets logging level to DEBUG (default).
        """
        if tf == True:
            cls.__logger.setLevel(level=logging.ERROR)
        else:
            cls.__logger.setLevel(level=logging.DEBUG)
