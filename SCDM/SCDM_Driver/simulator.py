import numpy as np
import uuid
from typing import List, Union, Dict
from utils import LoggingColor
from SCDM_Driver import SCDM_DriverInterface
from SCDM_Driver._virtual_matrix import VirtualMatrix

class SCDM_Simulator(SCDM_DriverInterface):
    """
    SCDM_Simulator
    ===============================================
    負責管理多個 VirtualMatrix 實例。
    
    [關於 CNN Kernel 的處理]
    對於 CNN (Convolutional Neural Network)，Submit 進來的矩陣應當是已經處理過的權重矩陣。
    通常是 (Out_Channels, In_Channels * Kernel_H * Kernel_W) 的形狀。
    Simulator 負責的是單純的矩陣乘法 (Matrix Multiplication)，
    im2col 或其他 Tensor 展開的操作應由上層處理完畢後傳入。

    ===============================================
    """

    def __init__(self, hw_rows=64, hw_cols=64):
        self.logger = LoggingColor.get_logger("SCDM_Simulator")
        self.hw_rows = hw_rows
        self.hw_cols = hw_cols
        
        # Structure: { id_str: VirtualMatrixObject }
        self.virtual_matrices: Dict[str, VirtualMatrix] = {}
        
        self._init_stats()
        
        self.logger.info(f"Initialized with HW Size: {hw_rows}x{hw_cols}")

    def _init_stats(self):
        """初始化統計數據字典"""
        # 保留當前 active 的 tile 數量，避免 reset 時狀態不一致
        current_active = 0
        if hasattr(self, 'stats'):
            current_active = self.stats.get("active_tiles", 0)

        self.stats = {
            # --- 邏輯層 (Logical) 統計 ---
            "logical_program_count": 0,    # submit 呼叫次數
            "logical_binary_ops": 0,       # compute_binary 呼叫次數
            "logical_multibit_ops": 0,     # compute_multibit 呼叫次數
            
            # --- 物理層 (Physical) 統計 - 更精確的功耗評估用 ---
            "physical_tiles_programmed": 0, # 總共對多少個硬體 Tile 進行了寫入
            "physical_binary_ops": 0,       # 實際硬體執行 binary 的總次數 (logical * tiles)
            "physical_multibit_ops": 0,     # 實際硬體執行 multibit 的總次數 (logical * tiles)
            
            # --- 資源狀態 ---
            "active_tiles": current_active, # 當前佔用的硬體 Tile 總數
            "total_tiles_created": 0        # 歷史總共創建過的 Tile 數
        }

    def submit_matrix(self, matrixes: List[np.ndarray]) -> str:
        """
        將一組矩陣寫入模擬器。
        
        Args:
            matrixes: 權重矩陣列表。
                      目前 Simulator 架構設計為一次處理一個邏輯層 (Layer)，
                      因此主要處理 matrixes[0]。
                      若為 CNN，請確保傳入前已 Flatten 為 2D Matrix。
        """
        group_id = str(uuid.uuid4())[:8]
        
        # 取出主權重矩陣
        target_matrix = matrixes[0]
        
        # 創建虛擬矩陣 (負責切割與分配硬體)
        v_matrix = VirtualMatrix(target_matrix, self.hw_rows, self.hw_cols)
        
        # 更新統計數據
        self.stats["logical_program_count"] += 1
        self.stats["physical_tiles_programmed"] += v_matrix.total_tiles
        self.stats["active_tiles"] += v_matrix.total_tiles
        self.stats["total_tiles_created"] += v_matrix.total_tiles
        
        self.virtual_matrices[group_id] = v_matrix
        
        self.logger.info(f"Matrix Submitted. ID: {group_id}, Tiles Used: {v_matrix.total_tiles}")
        return group_id

    def clear_matrix(self, id: str) -> bool:
        if id in self.virtual_matrices:
            v_matrix = self.virtual_matrices.pop(id)
            self.stats["active_tiles"] -= v_matrix.total_tiles
            self.logger.info(f"Matrix {id} cleared. Resources freed.")
            return True
        else:
            self.logger.warning(f"Attempted to clear non-existent ID: {id}")
            return False

    def clear_all_matrix(self) -> bool:
        cleared_count = len(self.virtual_matrices)
        self.virtual_matrices.clear()
        self.stats["active_tiles"] = 0
        self.logger.info(f"All matrices cleared ({cleared_count} groups).")
        return True
    
    def _execute_loop(self, id: str, input_data: np.ndarray, mode: str, bit_depth: int = 8) -> np.ndarray:
        """
        處理 Flatten -> Loop -> Reshape
        """
        if id not in self.virtual_matrices:
            raise ValueError(f"Matrix ID {id} not found.")
        
        v_matrix = self.virtual_matrices[id]
        
        # 1. 儲存原始形狀與特徵維度
        original_shape = input_data.shape
        input_features = original_shape[-1]
        
        if input_features > v_matrix.orig_rows:
            raise ValueError(f"Input features {input_features} > Matrix In_Features {v_matrix.orig_rows}")

        # 2. 攤平輸入 (Flatten) -> (N, In_Features)
        flat_input = input_data.reshape(-1, input_features)
        total_vectors = flat_input.shape[0]
        
        # 3. 準備輸出緩衝區 (N, Out_Features)
        flat_output = np.zeros((total_vectors, v_matrix.orig_cols), dtype=np.int32)
        
        # 4. 迴圈執行 (Vector by Vector)
        for i in range(total_vectors):
            vector = flat_input[i] # 這是嚴格的 1D Array
            
            # 呼叫 VirtualMatrix (它現在只吃 1D)
            flat_output[i] = v_matrix.compute(vector, mode=mode, bit_depth=bit_depth)
            
            # 更新統計 (以向量為單位)
            if mode == VirtualMatrix.MODE_BINARY:
                self.stats["logical_binary_ops"] += 1
                self.stats["physical_binary_ops"] += v_matrix.total_tiles
            else:
                self.stats["logical_multibit_ops"] += 1
                self.stats["physical_multibit_ops"] += v_matrix.total_tiles

        # 5. 還原形狀 (Reshape) -> (Batch, Seq, Out_Features)
        final_output_shape = original_shape[:-1] + (v_matrix.orig_cols,)
        return flat_output.reshape(final_output_shape)

    def compute_binary(self, id: str, input_vector: np.ndarray) -> np.ndarray:
        # 轉發給內部 Loop 處理
        return self._execute_loop(id, input_vector, mode=VirtualMatrix.MODE_BINARY)

    def compute_multibit(self, id: str, input_data: np.ndarray, bit_depth: int) -> np.ndarray:
        # 轉發給內部 Loop 處理
        return self._execute_loop(id, input_data, mode=VirtualMatrix.MODE_MULTIBIT, bit_depth=bit_depth)

    def get_statistic(self, id: Union[str, None] = None) -> dict:
        """
        回傳詳細統計數據。
        
        physical_* 數據可用於估算真實功耗:
        Total Energy ~= (physical_binary_ops * E_bin) + (physical_multibit_ops * E_multi)
        """
        # TODO 針對單一 ID 回傳統計
        return self.stats
    
    def reset_statistic(self) -> bool:
        """
        重置累計統計數據 (Ops, Program Count)，但保留當前硬體佔用狀態。
        """
        self.logger.info("Statistics Reset")
        
        # 保留當前 active tiles 數值
        current_active = self.stats["active_tiles"]
        
        self._init_stats()
        
        # 恢復 active tiles
        self.stats["active_tiles"] = current_active
        return True
