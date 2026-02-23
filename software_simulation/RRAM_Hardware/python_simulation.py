import numpy as np
from RRAM_Hardware import RRAM_HardwareInterface
from RRAM_Hardware.hardware_primitive import RRAM_HardwarePrimitive
from utils import LoggingColor

class RRAM_HardwarePythonSim(RRAM_HardwareInterface):
    """
    Filament Competitive RRAM Hardware Python Simulation
    
    這是一個通用的存算一體矩陣運算加速器封裝層。
    它嚴格遵守硬體定義的介面操作，將底層的類比物理行為 (電壓、電流、時間) 
    轉譯為標準的數位介面，並提供 Software-Defined Precision 的彈性。
    """
    
    # 初始化 Logger
    _logger = LoggingColor.get_logger("RRAM_HardwarePythonSim")

    def __init__(self, rows=64, cols=64):
        self.rows = rows
        self.cols = cols

        # 1. 初始化底層物理核心
        self.phy = RRAM_HardwarePrimitive(rows, cols)
        
        # 2. 影子暫存器 (Shadow Register)
        # 用於記錄當前陣列中的邏輯權重，以便在校正流程(Calibration)後恢復狀態。
        self._stored_matrix = np.zeros((self.rows, self.cols), dtype=int)
        
        # 3. 操作時序參數 (Timing Parameters)
        self.dt_binary = 0.8     # 二值模式積分時間 (0.8ns) - 快速
        self.dt_temporal = 10.0  # 時域模式積分時間 (10ns) - 高精度
        
        # 4. 類比參數 (Analog Parameters)
        self.v_threshold = 0.05     # 比較器閾值電壓
        self.conversion_factor = 0.0 # TDC (Time-to-Digital) 轉換係數 K

        # 5. 系統啟動校正 (System Startup Calibration)
        self._logger.info(LoggingColor.color_text(f"System Startup: Initializing ({self.rows}x{self.cols})...", LoggingColor.BOLD))
        self._perform_calibration()
        self._logger.info(LoggingColor.color_text("RRAM_HardwarePythonSim Ready.", LoggingColor.GREEN))

    def _perform_calibration(self):
        """ [內部程序] 系統校正：建立時間與強度的轉換係數 K """
        self._logger.debug("Starting TDC Calibration...")
        
        # Step 1: 抹除
        self.phy.reset_weights_physics()
        
        # Step 2: 寫入標準測試圖樣 (全1矩陣)
        calibration_pattern = np.ones((self.rows, self.cols), dtype=int)
        self.phy.program_weights_physics(calibration_pattern)
        
        # Step 3: 動態決定測試強度 (使用 Row 數的一半作為標準強度，避免溢位或過小)
        # 確保至少有 1 條線被驅動
        calib_strength = max(1, self.rows // 2)
        
        test_vector = np.zeros(self.rows)
        test_vector[:calib_strength] = 1 
        
        self.phy.reset_capacitors()
        self.phy.set_inputs_driver(test_vector)
        self.phy.run_physics_step(self.dt_temporal)
        
        triggered, times = self.phy.sense_spikes()
        
        # 檢查第 0 column (或其他任意 column) 是否有擊發
        if triggered[0]:
            # K = Value * Time
            # 這裡的 Value 就是我們剛剛送進去的強度 calib_strength
            self.conversion_factor = calib_strength * times[0]
            self._logger.debug(f"Calibration successful (Strength={calib_strength}). K-Factor: {self.conversion_factor:.2e}")
        else:
            # Fallback: 若未觸發，使用理論估計值
            self.conversion_factor = calib_strength * (self.dt_temporal * 0.5)
            self._logger.warning(LoggingColor.color_text("Calibration Timeout! Using fallback factor.", LoggingColor.YELLOW))

        # Step 4: 恢復使用者原本的矩陣資料
        self.phy.reset_weights_physics()
        if np.any(self._stored_matrix):
            self._logger.info("Restoring context from shadow register...")
            self.phy.program_weights_physics(self._stored_matrix)

    def _time_to_digital(self, times, trigger_mask):
        """ [內部程序] 查表轉換：將時間轉回數值 (Value = K / Time) """
        safe_times = np.maximum(times, 1e-15)
        estimated_values = np.round(self.conversion_factor / safe_times)
        return (estimated_values * trigger_mask).astype(int)

    # ==========================================================
    # 使用者指令集 (User Instruction Set)
    # ==========================================================

    def reset_matrix(self):
        """
        [指令] 矩陣重置 (Matrix Reset)
        
        對硬體陣列執行全區塊抹除 (Block Erase)，並同步清空軟體的影子暫存器。
        執行後，所有權重將回到高阻態 (邏輯上的 0 或初始態)。
        """
        self._logger.info("Command: Reset Matrix")
        
        # 1. 物理層抹除
        self.phy.reset_weights_physics()
        
        # 2. 清空影子暫存器
        self._stored_matrix.fill(0)

    def program_matrix(self, matrix_data):
        """
        [指令] 寫入矩陣 (Program Matrix)
        
        將權重資料寫入硬體陣列。
        此操作會同時更新內部的影子暫存器 (Shadow Register) 以支援後續的校正流程。

        Args:
            matrix_data (np.ndarray): 形狀為 (rows, cols) 的 Numpy 陣列。
                                      數值必須限制在 {-1, 0, 1}。
        Raises:
            ValueError: 若輸入矩陣尺寸不符。
        """
        expected_shape = (self.rows, self.cols)
        if matrix_data.shape != expected_shape:
            self._logger.error(LoggingColor.color_text(f"Matrix dimension mismatch! Expected {expected_shape}, got {matrix_data.shape}", LoggingColor.ERROR))
            raise ValueError(f"Matrix dimension must be {expected_shape}")
            
        self._logger.info(f"Command: Program Matrix (checksum: {np.sum(np.abs(matrix_data))})")
        
        # 1. 更新影子暫存器
        self._stored_matrix = matrix_data.astype(int)
        
        # 2. 執行物理層寫入
        self.phy.program_weights_physics(self._stored_matrix)

    def compute_binary(self, input_vector):
        """
        [指令] 二值運算 (Binary Compute / Threshold Mode)
        
        執行快速的向量-矩陣乘法，適用於二值化神經網路 (BNN) 或邏輯運算。

        Args:
            input_vector (np.ndarray): 長度為 rows 的一維陣列。
                                       數值應為 {-1, 0, 1}。

        Returns:
            np.ndarray: 長度為 cols 的一維整數陣列。
        """
        # 1. 重置電容
        self.phy.reset_capacitors()
        
        # 2. 設定輸入並短時間積分
        # 注意: Primitive 的 set_inputs_driver 預設需要符合 rows 長度的向量
        self.phy.set_inputs_driver(input_vector)
        self.phy.run_physics_step(self.dt_binary)
        
        # 3. 讀取類比電壓
        v_analog = self.phy.sense_voltage()
        
        # 4. 數位化判決 (Comparator Logic)
        output = np.zeros(self.cols, dtype=int)
        output[v_analog > self.v_threshold] = 1
        output[v_analog < -self.v_threshold] = -1
        
        return output

    def compute_multibit(self, input_data, bit_depth=8):
        """
        [指令] 多位元運算 (Multi-bit Compute / Temporal Mode)
        
        執行高精度的向量-矩陣乘法 (VMM)。
        本指令採用位元序列 (Bit-Serial) 架構，配合硬體的時域積分 (Temporal Integration) 模式。

        Args:
            input_data (np.ndarray): 長度為 rows 的一維整數陣列。
            bit_depth (int): 運算的位元深度 (預設為 8)。

        Returns:
            np.ndarray: 長度為 cols 的一維整數陣列。
        """
        final_output = np.zeros(self.cols, dtype=int)
        
        pos_mask = input_data >= 0
        neg_mask = input_data < 0
        
        # 使用 int 直接操作 (2's complement)
        raw_input = input_data.astype(int)

        # Bit-Serial Processing
        for b in range(bit_depth):
            # 1. Bit-Slicing
            bit_val = (raw_input >> b) & 1
            
            # 2. Driver Mapping
            hw_driver = np.zeros(self.rows)
            hw_driver[pos_mask] = bit_val[pos_mask]       # Pos: 1->1, 0->0
            
            # Neg: 1->0, 0->-1
            mask_neg_1 = (neg_mask) & (bit_val == 1)
            mask_neg_0 = (neg_mask) & (bit_val == 0)
            hw_driver[mask_neg_1] = 0
            hw_driver[mask_neg_0] = -1
            
            # 3. 執行物理積分
            self.phy.reset_capacitors()
            self.phy.set_inputs_driver(hw_driver)
            self.phy.run_physics_step(self.dt_temporal)
            
            # 4. 讀取結果
            triggered, times = self.phy.sense_spikes()
            v_analog = self.phy.sense_voltage()
            
            # 5. 時間-數值轉換
            magnitude = self._time_to_digital(times, triggered)
            
            # 6. 極性處理 & 累加
            sign = np.sign(v_analog)
            sign[sign == 0] = 1 
            
            term = magnitude * sign * (2 ** b)
            final_output += term.astype(int)

        # Correction Cycle
        # 針對負數輸入，硬體再跑一個 Cycle 送入 -1，以抵銷 Off-by-One 誤差
        if np.any(neg_mask):
            # 1. 建立修正向量: 負數行送 -1，其他送 0
            correction_driver = np.zeros(self.rows)
            correction_driver[neg_mask] = -1
            
            # 2. 執行物理積分 (真實硬體操作)
            self.phy.reset_capacitors()
            self.phy.set_inputs_driver(correction_driver)
            self.phy.run_physics_step(self.dt_temporal)
            
            # 3. 讀取結果
            triggered, times = self.phy.sense_spikes()
            v_analog = self.phy.sense_voltage()
            
            # 4. 轉換與極性
            magnitude = self._time_to_digital(times, triggered)
            sign = np.sign(v_analog)
            sign[sign == 0] = 1
            
            # 5. 直接累加 (權重為 1)
            # 因為送的是 -1，物理積分會自動產生正確的負向修正值
            final_output += (magnitude * sign).astype(int)
            
        return final_output
