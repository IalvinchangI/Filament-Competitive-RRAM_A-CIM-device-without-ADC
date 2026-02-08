"""
SCDM_HardwareSimple.py
===========================================================================
SCDM 簡易邏輯模擬器 (Logical Behavioral Model)
版本: v1.1 (Input Validation Added)

[用途]
這個類別用於上層演算法開發與快速驗證。
不考慮所有物理層的耗時模擬 (積分、雜訊、IR Drop)，
改用 Numpy 矩陣乘法直接計算，但保留硬體的位元操作邏輯。

[保留的硬體邏輯]
1. Bit-Serial 輸入切片。
2. 特殊負數映射 (1->0, 0->-1).
3. 負數修正週期 (Correction Cycle).
===========================================================================
"""

import numpy as np
from SCDM_Hardware import SCDM_HardwareInterface
from utils import LoggingColor

class SCDM_HardwareSimple(SCDM_HardwareInterface):
    """
    SCDM 理想邏輯模擬器
    直接使用 np.dot 進行運算，用於驗證邏輯正確性。
    """

    # 初始化 Logger
    _logger = LoggingColor.get_logger("SCDM_HardwareSimple")

    def __init__(self, rows=64, cols=64):
        self.rows = rows
        self.cols = cols
        
        # 理想記憶體 (直接存 int)
        self.matrix = np.zeros((rows, cols), dtype=int)
        
        self._logger.info(f"Initialized SCDM_HardwareSimple ({rows}x{cols})")

    def reset_matrix(self):
        """
        [指令] 矩陣重置 (Matrix Reset)
        
        對硬體陣列執行全區塊抹除 (Block Erase)。
        執行後，所有權重將回到高阻態 (邏輯上的 0 或初始態)。
        """
        self._logger.info("[Simple] Reset Matrix")
        self.matrix.fill(0)

    def program_matrix(self, matrix_data):
        """
        [指令] 寫入矩陣 (Program Matrix)
        
        將權重資料寫入硬體陣列。

        Args:
            matrix_data (np.ndarray): 形狀為 (rows, cols) 的 Numpy 陣列。
                                      數值必須限制在 {-1, 0, 1}。
        Raises:
            ValueError: 若輸入矩陣尺寸不符或數值不合法。
        """
        # 1. 檢查維度
        if matrix_data.shape != (self.rows, self.cols):
            self._logger.error(LoggingColor.color_text(f"Dimension mismatch: expected {(self.rows, self.cols)}, got {matrix_data.shape}", LoggingColor.ERROR))
            raise ValueError(f"Dimension mismatch: expected {(self.rows, self.cols)}")
        
        # 2. 檢查數值合法性 {-1, 0, 1}
        # 使用 np.isin 進行快速遮罩檢查
        if not np.all(np.isin(matrix_data, [-1, 0, 1])):
            self._logger.error(LoggingColor.color_text("Invalid Matrix Values! Must be in {-1, 0, 1}", LoggingColor.ERROR))
            raise ValueError("Invalid Matrix Values! Must be in {-1, 0, 1}")

        self._logger.info(f"Program Matrix (Sum: {np.sum(np.abs(matrix_data))})")
        self.matrix = matrix_data.astype(int)

    def compute_binary(self, input_vector):
        """
        [指令] 二值運算 (Binary Compute / Threshold Mode)
        
        執行快速的向量-矩陣乘法，適用於二值化神經網路 (BNN) 或邏輯運算。

        Args:
            input_vector (np.ndarray): 長度為 rows 的一維陣列。
                                       數值必須限制在 {-1, 0, 1}。

        Returns:
            np.ndarray: 長度為 cols 的一維整數陣列。
        """
        # 檢查數值合法性 {-1, 0, 1}
        if not np.all(np.isin(input_vector, [-1, 0, 1])):
            self._logger.error(LoggingColor.color_text("Invalid Input Vector Values! Must be in {-1, 0, 1}", LoggingColor.ERROR))
            raise ValueError("Invalid Input Vector Values! Must be in {-1, 0, 1}")

        # 1. 理想矩陣乘法
        # input_vector: {-1, 0, 1}
        # matrix: {-1, 0, 1}
        ideal_sum = np.dot(input_vector, self.matrix)
        
        # 2. 理想比較器 (Threshold = 0)
        # >0 -> 1, <0 -> -1, =0 -> 0
        output = np.sign(ideal_sum).astype(int)
        
        return output

    def compute_multibit(self, input_data, bit_depth=8):
        """
        [指令] 多位元運算 (Multi-bit Compute / Temporal Mode)
        
        執行高精度的向量-矩陣乘法 (VMM)。
        本指令採用位元序列 (Bit-Serial) 架構，配合硬體的時域積分 (Temporal Integration) 模式。

        Args:
            input_data (np.ndarray): 長度為 rows 的一維整數陣列 (任意整數)。
            bit_depth (int): 運算的位元深度 (預設為 8)。

        Returns:
            np.ndarray: 長度為 cols 的一維整數陣列。
        """
        final_output = np.zeros(self.cols, dtype=int)
        
        pos_mask = input_data >= 0
        neg_mask = input_data < 0
        raw_input = input_data.astype(int)

        # --- Phase 1: Bit-Serial Processing ---
        for b in range(bit_depth):
            # 1. 取出 Bit
            bit_val = (raw_input >> b) & 1
            
            # 2. 建立 Driver (模擬硬體的特殊映射邏輯)
            hw_driver = np.zeros(self.rows)
            
            # 正數: 1->1, 0->0
            hw_driver[pos_mask] = bit_val[pos_mask]
            
            # 負數: 1->0, 0->-1
            mask_neg_1 = (neg_mask) & (bit_val == 1)
            mask_neg_0 = (neg_mask) & (bit_val == 0)
            hw_driver[mask_neg_1] = 0
            hw_driver[mask_neg_0] = -1
            
            # 3. 理想運算 (取代物理積分)
            # 這裡用 np.dot 模擬硬體在該 Cycle 算出的 Partial Sum
            cycle_result = np.dot(hw_driver, self.matrix).astype(int)
            
            # 4. 移位累加
            final_output += cycle_result * (2 ** b)

        # --- Phase 2: Correction Cycle ---
        # 針對負數輸入，模擬硬體多跑一個 Cycle (送入 -1)
        if np.any(neg_mask):
            correction_driver = np.zeros(self.rows)
            correction_driver[neg_mask] = -1
            
            # 執行修正運算
            correction_result = np.dot(correction_driver, self.matrix).astype(int)
            
            # 累加 (權重為 1)
            final_output += correction_result

        return final_output.astype(int)
