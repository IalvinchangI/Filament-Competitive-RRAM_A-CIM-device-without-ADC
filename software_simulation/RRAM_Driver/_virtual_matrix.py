import math
import numpy as np
from typing import List, Tuple, Dict
from RRAM_Hardware import RRAM_HardwareSimple as Hardware
from utils import LoggingColor
import logging

class VirtualMatrix():
    """
    VirtualMatrix
    ===============================================
    負責將一個巨大的邏輯矩陣 (Logical Matrix) 切割並映射到多個
    物理硬體 (SCDM_HardwareSimple) 上。
    
    職責：
    1. Tiling (Padding -> Slicing)
    2. 管理多個 Hardware 實例
    3. 執行 Scatter-Gather (分發輸入 -> 收集輸出 -> 累加)

    ===============================================
    """
    
    # Mode Constants
    MODE_MULTIBIT = "multibit"

    # Statistic Key
    STATISTIC_WEIGHT_NEG1_KEY = "weight_neg1"
    STATISTIC_WEIGHT_0_KEY    = "weight_0"
    STATISTIC_WEIGHT_1_KEY    = "weight_1"
    STATISTIC_INPUT_NEG_KEY   = "input_neg"
    STATISTIC_INPUT_0_KEY     = "input_0"
    STATISTIC_INPUT_POS_KEY   = "input_pos"

    def __init__(self, full_matrix: np.ndarray, hw_rows: int, hw_cols: int, ideal_TF: bool = False, silent_hardware: bool = True):
        self.logger = LoggingColor.get_logger("VirtualMatrix")
        if silent_hardware:
            Hardware._logger.setLevel(logging.WARNING)
        
        self.orig_rows, self.orig_cols = full_matrix.shape
        self.hw_rows = hw_rows
        self.hw_cols = hw_cols
        self.ideal_TF = ideal_TF
        
        # 計算 Grid 大小 (無條件進位)
        self.grid_rows = math.ceil(self.orig_rows / hw_rows)
        self.grid_cols = math.ceil(self.orig_cols / hw_cols)
        self.total_tiles = self.grid_rows * self.grid_cols
        
        # 建立硬體網格: List of Lists storing SCDM_HardwareSimple
        self.tiles: List[List[Hardware]] = []

        self.weight_stats: Dict[str, int] = dict()
        
        # 執行切割與燒錄
        self._tiling_and_program(full_matrix)

    def _tiling_and_program(self, matrix: np.ndarray):
        """
        將大矩陣切割並寫入各個硬體 Tile
        """
        # 預先 Pad 矩陣以符合硬體倍數
        pad_r = self.grid_rows * self.hw_rows - self.orig_rows
        pad_c = self.grid_cols * self.hw_cols - self.orig_cols
        
        # 使用 0 填充 (不影響運算結果)
        padded_matrix = np.pad(matrix, ((0, pad_r), (0, pad_c)), mode='constant', constant_values=0)

        self.weight_stats = {
            self.STATISTIC_WEIGHT_NEG1_KEY: int(np.sum(padded_matrix == -1)), 
            self.STATISTIC_WEIGHT_0_KEY:    int(np.sum(padded_matrix == 0)), 
            self.STATISTIC_WEIGHT_1_KEY:    int(np.sum(padded_matrix == 1))
        }
        
        for r in range(self.grid_rows):
            row_tiles = []
            for c in range(self.grid_cols):
                # 1. 實例化硬體
                hw = Hardware(rows=self.hw_rows, cols=self.hw_cols, ideal_TF=self.ideal_TF)
                
                # 2. 切割子矩陣
                r_start = r * self.hw_rows
                r_end = r_start + self.hw_rows
                c_start = c * self.hw_cols
                c_end = c_start + self.hw_cols
                
                sub_matrix = padded_matrix[r_start:r_end, c_start:c_end]
                
                # 3. 寫入硬體
                hw.program_matrix(sub_matrix)
                row_tiles.append(hw)
            self.tiles.append(row_tiles)
            
        self.logger.info(f"Created VirtualMatrix: {self.orig_rows}x{self.orig_cols} "
                         f"mapped to {self.grid_rows}x{self.grid_cols} tiles.")

    def compute(self, input_vector: np.ndarray, mode: str, bit_depth: int = 8) -> Tuple[np.ndarray, Dict]:
        """
        執行分塊運算並組合結果。
        
        Returns:
            np.ndarray: Raw Sum (未經 Activation 的整數累加值)
            dict
        """
        # 1. Pad Input Vector
        pad_len = self.grid_rows * self.hw_rows - len(input_vector)
        if pad_len < 0:
             raise ValueError("Input vector too large for programmed matrix")
        padded_input = np.pad(input_vector, (0, pad_len), mode='constant', constant_values=0)

        input_stats = {
            self.STATISTIC_INPUT_NEG_KEY: int(np.sum(padded_input <= -1)) * self.grid_cols, 
            self.STATISTIC_INPUT_0_KEY:   int(np.sum(padded_input == 0)) * self.grid_cols, 
            self.STATISTIC_INPUT_POS_KEY: int(np.sum(padded_input >= 1)) * self.grid_cols
        }
        
        # 準備輸出緩衝區 (累加 Partial Sums 用，使用 int32 防止溢位)
        output_buffer = np.zeros(self.grid_cols * self.hw_cols, dtype=int)
        
        # 2. 雙重迴圈執行運算 (Row-wise reduce, Col-wise concat)
        for r in range(self.grid_rows):
            # 取出對應這列 Tiles 的輸入向量切片
            r_start = r * self.hw_rows
            r_end = r_start + self.hw_rows
            input_slice = padded_input[r_start:r_end]
            
            for c in range(self.grid_cols):
                hw_tile = self.tiles[r][c]
                
                # 呼叫硬體並取得 Partial Sum
                if mode == self.MODE_MULTIBIT:
                    input_slice_pos = np.zeros_like(input_slice, dtype=int)
                    input_slice_neg = np.zeros_like(input_slice, dtype=int)
                    input_slice_pos[input_slice >= 0] = input_slice[input_slice >= 0]
                    input_slice_neg[input_slice < 0] = -(input_slice[input_slice < 0])

                    tile_out_pos = hw_tile.compute_multibit(input_slice_pos, bit_depth=bit_depth)
                    tile_out_neg = hw_tile.compute_multibit(input_slice_neg, bit_depth=bit_depth)
                    tile_out = tile_out_pos - tile_out_neg
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                # 累加到對應的 Output Channel
                c_start = c * self.hw_cols
                c_end = c_start + self.hw_cols
                output_buffer[c_start:c_end] += tile_out

        # 3. 裁切掉 Padding 的輸出，直接回傳 Raw Sum
        final_output = output_buffer[:self.orig_cols]
            
        return final_output, input_stats
