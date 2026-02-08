import numpy

class SCDM_HardwareInterface():
    """
    SCDM 硬體抽象層介面 (Hardware Abstraction Layer Interface)
    定義所有 SCDM 實作 (Simulation / FPGA / ASIC) 必須遵守的標準 API。
    """

    # =====================================================================
    # 系統控制 (System Control)
    # =====================================================================

    def reset_matrix(self):
        """
        [指令] 矩陣重置 (Matrix Reset)
        對硬體陣列執行全區塊抹除，所有權重回歸初始態 (通常為 0 / High-Z)。
        """
        raise NotImplementedError()

    def program_matrix(self, matrix_data: numpy.ndarray):
        """
        [指令] 寫入矩陣 (Program Matrix)
        將權重資料寫入硬體陣列。

        Args:
            matrix_data: 2D numpy array, shape=(rows, cols).
                         數值必須限制在 {-1, 0, 1}。
        """
        raise NotImplementedError()
    
    # =====================================================================
    # 運算指令 (Compute Instructions)
    # =====================================================================

    def compute_binary(self, input_vector: numpy.ndarray) -> numpy.ndarray:
        """
        [指令] 二值運算 (Binary Compute / Threshold Mode)
        執行快速向量矩陣乘法，輸出僅含 {-1, 0, 1}。

        Args:
            input_vector: 1D numpy array, shape=(rows,). Values in {-1, 0, 1}.
        Returns:
            output_vector: 1D numpy array, shape=(cols,). Values in {-1, 0, 1}.
        """
        raise NotImplementedError()

    def compute_multibit(self, input_data: numpy.ndarray, bit_depth: int = 8) -> numpy.ndarray:
        """
        [指令] 多位元運算 (Multi-bit Compute / Temporal Mode)
        執行高精度向量矩陣乘法，支援軟體定義精度 (Software Defined Precision)。

        Args:
            input_data: 1D numpy array, shape=(rows,). 任意整數 (Integer)。
            bit_depth: 運算位元深度 (e.g., 4, 8, 16)。
        Returns:
            output_vector: 1D numpy array, shape=(cols,). 高精度整數結果。
        """
        raise NotImplementedError()
