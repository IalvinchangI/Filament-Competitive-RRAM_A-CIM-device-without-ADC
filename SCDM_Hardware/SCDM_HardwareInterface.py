import numpy
from typing import Tuple

class SCDM_HardwareInterface():

    # =====================================================================
    # 硬體指令：寫入與組態 (Programming)
    # =====================================================================

    def reset_weights_physics(self):
        raise NotImplementedError()

    def program_weights_physics(self, target_weights: numpy.ndarray):
        """
        Args:
            target_weights: {-1, 0, 1}
        """
        raise NotImplementedError()
    
    # =====================================================================
    # 硬體指令：執行期控制 (Runtime Control)
    # =====================================================================

    def set_inputs_driver(self, input_vector: numpy.ndarray):
        raise NotImplementedError()
    
    def reset_capacitors(self):
        raise NotImplementedError()
    
    def run_physics_step(self, duration_ns: int):
        raise NotImplementedError()
    
    # =====================================================================
    # 讀出介面 (Readout Interface)
    # =====================================================================

    def sense_voltage(self) -> numpy.ndarray:
        raise NotImplementedError()

    def sense_spikes(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        raise NotImplementedError()
