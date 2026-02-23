import numpy as np
import uuid
from copy import deepcopy
from typing import List, Union, Dict
from utils import LoggingColor
from RRAM_Driver import RRAM_DriverInterface
from RRAM_Driver._virtual_matrix import VirtualMatrix

class RRAM_Simulator(RRAM_DriverInterface):
    """
    RRAM Simulator
    ===========================================================================
    This class implements the RRAM_DriverInterface to manage multiple 
    VirtualMatrix instances in a simulated environment.

    Key Responsibilities:
    1. Virtualization Management: Creates and manages VirtualMatrix objects.
    2. Preprocessing Expectation: Expects incoming CNN weights to be pre-flattened 
       (e.g., Out_Channels, In_Channels * Kernel_H * Kernel_W).
    3. Execution: Orchestrates matrix multiplication computations without handling 
       tensor unfolding (like im2col).
    4. Statistics Tracking: Maintains detailed logical and physical operation 
       statistics for power and performance estimation.
    
    ===========================================================================
    """

    def __init__(self, hw_rows=256, hw_cols=256, ideal_TF: bool = False):
        self.logger = LoggingColor.get_logger("RRAM_Simulator")
        self.hw_rows = hw_rows
        self.hw_cols = hw_cols
        self.ideal_TF = ideal_TF
        
        # Structure: { id_str: VirtualMatrixObject }
        self.virtual_matrices: Dict[str, VirtualMatrix] = dict()
        # Structure: { id_str: per_id_stat_dict }
        self.per_id_stats: Dict[str, dict] = dict()
        
        self._init_stats()
        self.logger.info(f"Initialized with HW Size: {hw_rows}x{hw_cols}")

    def _init_stats(self):
        current_active = 0
        tiles_created = 0
        w_neg1 = 0
        w_0 = 0
        w_1 = 0
        
        if hasattr(self, 'stats'):
            current_active = self.stats.get("active_tiles", 0)
            tiles_created = self.stats.get("total_tiles_created", 0)
            w_neg1 = self.stats.get("total_programmed_weight_neg1", 0)
            w_0 = self.stats.get("total_programmed_weight_0", 0)
            w_1 = self.stats.get("total_programmed_weight_1", 0)

        self.stats = {
            "logical_program_count": 0,
            "logical_multibit_ops": 0,
            
            "physical_tiles_programmed": 0,
            "physical_multibit_ops": 0,
            
            "active_tiles": current_active,
            "total_tiles_created": tiles_created,

            "total_programmed_weight_neg1": w_neg1, 
            "total_programmed_weight_0": w_0, 
            "total_programmed_weight_1": w_1, 
            
            "total_computed_input_neg": 0, 
            "total_computed_input_0": 0, 
            "total_computed_input_pos": 0, 
        }

        if hasattr(self, 'per_id_stats'):
            for gid, stat in self.per_id_stats.items():
                stat["logical_multibit_ops"] = 0
                stat["physical_multibit_ops"] = 0
                stat["computed_input_neg"] = 0
                stat["computed_input_0"] = 0
                stat["computed_input_pos"] = 0

    def submit_matrix(self, matrixes: List[np.ndarray]) -> str:
        """
        Register and program a list of matrices into the simulator.

        Creates a VirtualMatrix to handle tiling and hardware mapping. Currently 
        processes one logical layer at a time (primarily uses matrixes[0]).

        Args:
            matrixes (List[np.ndarray]): A list containing the weight matrix to be programmed.

        Returns:
            str: A unique Group ID representing this specific submitted matrix.

        Records:
            - Increments 'logical_program_count' and 'total_tiles_created'.
            - Updates 'physical_tiles_programmed' and 'active_tiles'.
            - Accumulates weight distribution statistics (-1, 0, 1).
        """
        group_id = str(uuid.uuid4())[:8]
        target_matrix = matrixes[0]
        
        v_matrix = VirtualMatrix(target_matrix, self.hw_rows, self.hw_cols)
        
        self.stats["logical_program_count"] += 1
        self.stats["physical_tiles_programmed"] += v_matrix.total_tiles
        self.stats["active_tiles"] += v_matrix.total_tiles
        self.stats["total_tiles_created"] += v_matrix.total_tiles
        self.stats["total_programmed_weight_neg1"] += v_matrix.weight_stats[VirtualMatrix.STATISTIC_WEIGHT_NEG1_KEY]
        self.stats["total_programmed_weight_0"] += v_matrix.weight_stats[VirtualMatrix.STATISTIC_WEIGHT_0_KEY]
        self.stats["total_programmed_weight_1"] += v_matrix.weight_stats[VirtualMatrix.STATISTIC_WEIGHT_1_KEY]
        
        self.per_id_stats[group_id] = {
            "original_shape": [v_matrix.orig_rows, v_matrix.orig_cols], 
            "physical_tiles": v_matrix.total_tiles, 
            "weight_stats": v_matrix.weight_stats.copy(), 
            "logical_multibit_ops": 0, 
            "physical_multibit_ops": 0, 
            "computed_input_neg": 0, 
            "computed_input_0": 0, 
            "computed_input_pos": 0, 
        }

        self.virtual_matrices[group_id] = v_matrix
        
        self.logger.info(f"Matrix Submitted. ID: {group_id}, Tiles Used: {v_matrix.total_tiles}")
        return group_id

    def clear_matrix(self, id: str) -> bool:
        """
        Release the hardware resources associated with a specific Group ID.

        Frees up the simulated hardware tiles and updates global statistics.

        Args:
            id (str): The unique Group ID returned by `submit_matrix`.

        Returns:
            bool: True if the ID existed and was successfully cleared, False otherwise.

        Records:
            - Decrements 'active_tiles'.
            - Subtracts the specific matrix's weight distribution from global statistics.
        """
        if id in self.virtual_matrices:
            v_matrix = self.virtual_matrices.pop(id)
            
            self.stats["active_tiles"] -= v_matrix.total_tiles
            self.stats["total_programmed_weight_neg1"] -= v_matrix.weight_stats[VirtualMatrix.STATISTIC_WEIGHT_NEG1_KEY]
            self.stats["total_programmed_weight_0"] -= v_matrix.weight_stats[VirtualMatrix.STATISTIC_WEIGHT_0_KEY]
            self.stats["total_programmed_weight_1"] -= v_matrix.weight_stats[VirtualMatrix.STATISTIC_WEIGHT_1_KEY]
            
            if id in self.per_id_stats:
                del self.per_id_stats[id]
                
            self.logger.info(f"Matrix {id} cleared. Resources freed.")
            return True
        else:
            self.logger.warning(f"Attempted to clear non-existent ID: {id}")
            return False

    def clear_all_matrix(self) -> bool:
        """
        Global Reset.

        Clears all submitted matrices, releases all simulated hardware resources, 
        and resets static resource counters.

        Returns:
            bool: True indicating the reset was successful.

        Records:
            - Resets 'active_tiles' and all static weight statistics to 0.
        """
        cleared_count = len(self.virtual_matrices)
        self.virtual_matrices.clear()
        self.per_id_stats.clear()
        
        self.stats["active_tiles"] = 0
        self.stats["total_programmed_weight_neg1"] = 0
        self.stats["total_programmed_weight_0"] = 0
        self.stats["total_programmed_weight_1"] = 0
        
        self.logger.info(f"All matrices cleared ({cleared_count} groups).")
        return True

    def compute_multibit(self, id: str, input_data: np.ndarray, bit_depth: int) -> np.ndarray:
        """
        Perform a Multi-bit Computation on the specified matrix group.

        Flattens the input if necessary, processes vector by vector through the 
        underlying VirtualMatrix, and reshapes the output to match the expected batch/sequence.

        Args:
            id (str): The Group ID to compute against.
            input_data (np.ndarray): The multi-bit input array. The last dimension 
                                     must match the matrix's input features.
            bit_depth (int): The precision (number of bits) for the computation.

        Returns:
            np.ndarray: The result array of high precision integers.

        Raises:
            ValueError: If the ID is not found or if input features exceed matrix capacity.

        Records:
            - Increments 'logical_multibit_ops' and 'physical_multibit_ops'.
            - Accumulates computed input statistics (negative, zero, positive).
        """
        if id not in self.virtual_matrices:
            raise ValueError(f"Matrix ID {id} not found.")
        
        v_matrix = self.virtual_matrices[id]
        id_stat = self.per_id_stats[id] 
        
        original_shape = input_data.shape
        input_features = original_shape[-1]
        
        if input_features > v_matrix.orig_rows:
            raise ValueError(f"Input features {input_features} > Matrix In_Features {v_matrix.orig_rows}")

        flat_input = input_data.reshape(-1, input_features)
        total_vectors = flat_input.shape[0]
        
        flat_output = np.zeros((total_vectors, v_matrix.orig_cols), dtype=np.int32)
        
        for i in range(total_vectors):
            vector = flat_input[i]
            flat_output[i], in_stats = v_matrix.compute(vector, mode=VirtualMatrix.MODE_MULTIBIT, bit_depth=bit_depth)
            
            self.stats["logical_multibit_ops"] += 1
            self.stats["physical_multibit_ops"] += v_matrix.total_tiles * 2
            
            id_stat["logical_multibit_ops"] += 1
            id_stat["physical_multibit_ops"] += v_matrix.total_tiles * 2
                
            self.stats["total_computed_input_neg"] += in_stats[VirtualMatrix.STATISTIC_INPUT_NEG_KEY]
            self.stats["total_computed_input_0"] += in_stats[VirtualMatrix.STATISTIC_INPUT_0_KEY]
            self.stats["total_computed_input_pos"] += in_stats[VirtualMatrix.STATISTIC_INPUT_POS_KEY]
            
            id_stat["computed_input_neg"] += in_stats[VirtualMatrix.STATISTIC_INPUT_NEG_KEY]
            id_stat["computed_input_0"] += in_stats[VirtualMatrix.STATISTIC_INPUT_0_KEY]
            id_stat["computed_input_pos"] += in_stats[VirtualMatrix.STATISTIC_INPUT_POS_KEY]

        final_output_shape = original_shape[:-1] + (v_matrix.orig_cols,)
        return flat_output.reshape(final_output_shape)

    def get_statistic(self, id: Union[str, None] = None) -> dict:
        """
        Retrieve performance and usage statistics for power/performance modeling.

        Args:
            id (Union[str, None], optional): 
                - If None: Returns a dictionary containing all global and per-ID stats.
                - If "": Returns only global statistics.
                - If specific ID: Returns statistics relevant to that matrix group.

        Returns:
            dict: A dictionary containing the requested statistics.
        """
        if id is None:
            return {
                "stats": deepcopy(self.stats), 
                "per_id_stats": deepcopy(self.per_id_stats)
            }

        if id == "":
            return self.stats
        
        if id not in self.per_id_stats:
            self.logger.warning(f"Statistic for ID '{id}' not found.")
            return {}
            
        return self.per_id_stats[id]
    
    def reset_statistic(self) -> bool:
        """
        Reset the cumulative statistics (Ops, Input counts) to zero.

        Retains the current hardware occupancy status and weight distribution.

        Returns:
            bool: True indicating the reset was successful.
        """
        self.logger.info("Statistics Reset")
        self._init_stats()
        return True
