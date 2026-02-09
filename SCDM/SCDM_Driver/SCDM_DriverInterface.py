import numpy
from typing import List, Union, Dict

class SCDM_DriverInterface():
    """
    SCDM Driver Interface
    ===========================================================================
    This class serves as an abstraction layer (HAL) between high-level algorithms
    and the low-level SCDM hardware simulators.

    Key Responsibilities:
    1. Resource Management: Pools and manages multiple SCDM_Hardware instances.
    2. Virtualization: Maps logical matrices (User view) to physical hardware tiles (Hardware view).
    3. Auto-Tiling: Automatically splits matrices larger than the hardware dimensions.
    4. Performance Monitoring: Tracks usage statistics (Ops, Cycles, Program Counts) for power estimation.
    ===========================================================================
    """

    def submit_matrix(self, matrixes: List[numpy.ndarray]) -> str:
        """
        Register and program a list of matrices into the SCDM hardware.

        This function handles the mapping from logical matrices to physical hardware.
        If a matrix is larger than the underlying hardware dimensions (rows/cols),
        the driver will automatically perform 'Tiling' (split the matrix) and assign
        it to multiple hardware instances.

        Args:
            matrixes (List[numpy.ndarray]): A list of 2D numpy arrays (weights) to be programmed.
                                            Values should generally be within the hardware's supported range (e.g., -1, 0, 1).

        Returns:
            str: A unique 'Group ID' (Handle) representing this specific set of matrices.
                 The user will use this ID for subsequent computation requests.
                 All hardware tiles associated with this ID will operate logically as a single unit.

        Records:
            - Increments the 'program_count' statistic.
            - Updates 'active_tiles' usage.
        """
        raise NotImplementedError()
    
    def clear_matrix(self, id: str) -> bool:
        """
        Release the hardware resources associated with a specific Group ID.
        
        This frees up the SCDM hardware tiles for other tasks.

        Args:
            id (str): The unique Group ID returned by `submit_matrix`.

        Returns:
            bool: True if the ID existed and was successfully cleared, False otherwise.

        Records:
            - Decrements 'active_tiles' usage.
        """
        raise NotImplementedError()
    
    def clear_all_matrix(self) -> bool:
        """
        Global Reset.
        
        Clears all submitted matrices and releases all hardware resources.
        Essentially restores the driver to its initial state.

        Returns:
            bool: True if reset was successful.
        """
        raise NotImplementedError()
    
    def compute_binary(self, id: str, input_vector: numpy.ndarray) -> numpy.ndarray:
        """
        Perform a Binary Computation (XNOR/Popcount or similar) on the specified matrix group.

        If the matrix was tiled across multiple hardware units, the driver is responsible for:
        1. Broadcasting the input vector to relevant tiles.
        2. Accumulating partial sums from multiple tiles.
        3. Applying the final threshold function (Sign).

        Args:
            id (str): The Group ID to compute against.
            input_vector (numpy.ndarray): The input activation vector.

        Returns:
            numpy.ndarray: The result vector (typically -1, 0, 1 for ternary/binary logic).

        Records:
            - Increments 'compute_binary_ops'.
        """
        raise NotImplementedError()
    
    def compute_multibit(self, id: str, input_data: numpy.ndarray, bit_depth: int) -> numpy.ndarray:
        """
        Perform a Multi-bit Computation (MAC operation) on the specified matrix group.

        This simulates the temporal integration or bit-serial processing of the hardware.

        Args:
            id (str): The Group ID to compute against.
            input_data (numpy.ndarray): The multi-bit input vector.
            bit_depth (int): The precision (number of bits) for the input/computation.

        Returns:
            numpy.ndarray: The result vector (High precision integers).

        Records:
            - Increments 'compute_multibit_ops'.
        """
        raise NotImplementedError()

    def get_statistic(self, id: Union[str, None] = None) -> Dict:
        """
        Retrieve performance and usage statistics for power/performance modeling.

        Statistics typically include:
        - Program Counts (Writing weights)
        - Compute Operations (Binary vs. Multibit)
        - Active Hardware Tiles
        - Estimated Cycle Counts

        Args:
            id (Union[str, None]): 
                - If None: Returns global statistics (all time).
                - If specific ID: Returns statistics relevant to that matrix group.
        
        Returns:
            dict: A dictionary containing key-value pairs of the requested statistics.
        """
        raise NotImplementedError()
    
    def reset_statistic(self) -> bool:
        """
        Reset the cumulative statistics (Ops, Program counts, etc.) to zero.
        
        This is useful for benchmarking specific phases (e.g., reset after model loading
        to measure only the inference energy).
        
        Note: Status indicators (like 'active_tiles') should reflect the current 
        hardware state and might not be zeroed if matrices are still loaded.

        Returns:
            bool: True if reset was successful.
        """
        raise NotImplementedError()