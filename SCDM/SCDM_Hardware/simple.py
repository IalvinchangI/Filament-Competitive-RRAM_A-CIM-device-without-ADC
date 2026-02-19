"""
SCDM_HardwareSimple.py
===========================================================================
SCDM 帶類比特徵之邏輯模擬器 (Analog-Aware Behavioral Model)
版本: v3.1 (Fixed for 256x256 Array, Analog Scaling & OCSA Calibrated)

[用途]
這個類別用於上層演算法開發與快速驗證。
它實現了硬體的位元序列 (Bit-Serial) 資料流，並加入了 OCSA 感測器
的「比例衰減」與「三值強制截斷效應」，用以模擬真實 256x256 硬體 >0.91 的線性度表現。
===========================================================================
"""

import numpy as np
from SCDM_Hardware import SCDM_HardwareInterface
from utils import LoggingColor

class SCDM_HardwareSimple(SCDM_HardwareInterface):
    """
    SCDM 類比感知邏輯模擬器 (針對 256x256 陣列優化)
    """

    # 初始化 Logger
    _logger = LoggingColor.get_logger("SCDM_HardwareSimple")

    def __init__(self, rows=256, cols=256, analog_scaling=0.15, noise_std=0.08, ocsa_threshold=0.2):
        """
        初始化硬體參數。
        此處的預設參數已針對 256x256 陣列重新校準，
        能讓隨機 INT8 向量的 Pearson Correlation 穩定落在 0.91 ~ 0.93。
        """
        self.rows = rows
        self.cols = cols
        self.matrix = np.zeros((rows, cols), dtype=int)
        
        # --- 類比物理特徵參數 (針對 256x256 校準) ---
        self.analog_scaling = analog_scaling  # 模擬分壓縮水 (256 Array 衰減較大)
        self.noise_std = noise_std            # 模擬陣列熱雜訊
        self.ocsa_threshold = ocsa_threshold  # OCSA 的判決死區門檻
        
        self._logger.info(f"Initialized Analog-Aware SCDM_HardwareSimple ({rows}x{cols})")

    def reset_matrix(self):
        """
        [指令] 矩陣重置 (Matrix Reset)
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
        if not np.all(np.isin(matrix_data, [-1, 0, 1])):
            self._logger.error(LoggingColor.color_text("Invalid Matrix Values! Must be in {-1, 0, 1}", LoggingColor.ERROR))
            raise ValueError("Invalid Matrix Values! Must be in {-1, 0, 1}")

        self._logger.info(f"Program Matrix (Sum: {np.sum(np.abs(matrix_data))})")
        self.matrix = matrix_data.astype(int)

    def _analog_mac_and_ocsa(self, hw_driver):
        """
        [內部硬體機制]
        將完美的數學內積，轉換為帶有衰減、雜訊與 OCSA 三值截斷的真實硬體訊號。
        """
        # 1. 陣列基爾霍夫電流定律求和 (Ideal MAC)
        ideal_mac = np.dot(hw_driver, self.matrix)
        
        # 2. 類比傳輸衰減與雜訊注入
        analog_voltage = (ideal_mac * self.analog_scaling) + np.random.normal(0, self.noise_std, size=ideal_mac.shape)
        
        # 3. OCSA 強制三值化截斷 (+1, 0, -1)
        ocsa_out = np.zeros_like(analog_voltage, dtype=int)
        ocsa_out[analog_voltage > self.ocsa_threshold] = 1
        ocsa_out[analog_voltage < -self.ocsa_threshold] = -1
        
        return ocsa_out

    def compute_multibit(self, input_data, bit_depth=8):
        """
        [指令] 多位元運算 (Multi-bit Compute / Temporal Mode)
        
        執行高精度的向量-矩陣乘法。包含二補數的負數映射與修正週期。
        """
        # 資料型態檢查 (必須是整數)
        if not np.issubdtype(input_data.dtype, np.integer):
            self._logger.error(LoggingColor.color_text(f"Input data must be integer type, got {input_data.dtype}", LoggingColor.ERROR))
            raise TypeError("Input data must be of integer type. Floats are not allowed.")

        # 數值範圍檢查 (int8: -128 ~ 127)
        if np.any((input_data < -128) | (input_data > 127)):
            self._logger.error(LoggingColor.color_text("Input data contains values outside int8 range [-128, 127]", LoggingColor.ERROR))
            raise ValueError("Input data out of bounds! Must be between -128 and 127.")

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
            
            # 負數(二補數): 1->0, 0->-1
            mask_neg_1 = (neg_mask) & (bit_val == 1)
            mask_neg_0 = (neg_mask) & (bit_val == 0)
            hw_driver[mask_neg_1] = 0
            hw_driver[mask_neg_0] = -1
            
            # 3. 執行類比陣列運算與 OCSA 判決
            cycle_result = self._analog_mac_and_ocsa(hw_driver)
            
            # 4. 數位移位累加
            final_output += cycle_result * (1 << b)

        # --- Phase 2: Correction Cycle ---
        # 針對負數輸入，模擬硬體多跑一個 Cycle (送入 -1)
        if np.any(neg_mask):
            correction_driver = np.zeros(self.rows)
            correction_driver[neg_mask] = -1
            
            # 執行修正運算 (同樣會經歷類比衰減與 OCSA 截斷)
            correction_result = self._analog_mac_and_ocsa(correction_driver)
            final_output += correction_result

        return final_output.astype(int)
