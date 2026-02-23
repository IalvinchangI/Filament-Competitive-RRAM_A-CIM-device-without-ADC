import numpy

class RRAM_HardwareInterface():
    """
    RRAM Hardware Interface
    ===========================================================================
    This class serves as the Hardware Abstraction Layer (HAL) interface for 
    Filament Competitive RRAM. 

    Key Responsibilities:
    1. Standardization: Defines the standard API that all Filament Competitive 
       RRAM implementations (Simulation, FPGA, ASIC) must follow.
    2. Hardware Control: Provides low-level commands for resetting and programming.
    3. Computation: Exposes methods for multi-bit vector-matrix multiplication.
    
    ===========================================================================
    """

    # =====================================================================
    # Hardware Control
    # =====================================================================

    def reset_matrix(self):
        """
        Perform a global block erase on the hardware array.
        
        This resets all weights to their initial state (typically 0 or High-Z).

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError()

    def program_matrix(self, matrix_data: numpy.ndarray):
        """
        Program weight data into the hardware array.

        Args:
            matrix_data (numpy.ndarray): A 2D numpy array representing the weights, 
                                         shape=(rows, cols). Values must be restricted 
                                         to {-1, 0, 1}.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError()
    
    # =====================================================================
    # Compute Instructions
    # =====================================================================

    def compute_multibit(self, input_data: numpy.ndarray, bit_depth: int = 8) -> numpy.ndarray:
        """
        Perform high-precision vector-matrix multiplication (Temporal Mode).

        Executes multi-bit computation supporting software-defined precision.

        Args:
            input_data (numpy.ndarray): A 1D numpy array of integers, shape=(rows,).
            bit_depth (int, optional): The compute bit depth (e.g., 4, 8, 16). Defaults to 8.

        Returns:
            numpy.ndarray: A 1D numpy array of high-precision integer results, shape=(cols,).

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError()
