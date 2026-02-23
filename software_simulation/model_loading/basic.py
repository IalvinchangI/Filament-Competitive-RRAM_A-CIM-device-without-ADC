from RRAM_Driver import RRAM_DriverInterface
import numpy

class BasicModelLoader():
    """
    Basic Model Loader
    ===========================================================================
    This class serves as the abstract base class (interface) bridging trained 
    models and the ModelExecutor.

    Key Responsibilities:
    1. Model Parsing: Acts as the base template for parsing specific model architectures.
    2. Hardware Offloading: Defines and configures which specific layers or steps 
       can be offloaded to the Filament Competitive RRAM hardware for computation.
    
    ===========================================================================
    """

    PREDICT_COMMAND = "predict_command"

    def config(self, driver: RRAM_DriverInterface):
        """
        Configure the hardware driver for the loaded model.

        This method should map the model's weights to the hardware and set up 
        any necessary adapters for computation.

        Args:
            driver (RRAM_DriverInterface): The hardware driver instance to be used.
        """
        self.driver = driver

    def run(self, input_data: numpy.ndarray, input_command: str = None) -> numpy.ndarray:
        """
        Execute the model with the given input data.

        Args:
            input_data (numpy.ndarray): The input data for the model.
            input_command (str, optional): A command string defining the execution 
                                           mode (e.g., `PREDICT_COMMAND`). Defaults to None.

        Returns:
            numpy.ndarray: The output result from the model execution.
        """
        pass

    def unload(self):
        """
        Unload the model and release associated hardware resources.

        This should clear weights from the RRAM hardware and free up system memory.
        """
        pass

    def loader_details(self):
        """
        Retrieve specific configuration details of the model loader.

        Returns:
            dict: A dictionary containing loader-specific configuration parameters.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError
