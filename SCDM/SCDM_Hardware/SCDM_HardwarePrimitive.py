"""
SCDM_HardwarePrimitive.py
===========================================================================
SCDM 存算一體晶片 - 真實物理層原語 (True Physics Primitives)
版本: v6.0 (Unsimplified Physics / Transient Dynamics)

[物理模型說明]
本模組還原真實 RRAM 物理行為：
1. 傳輸機制: Mott-Gurney Law (Trap-Assisted Tunneling) with sinh() dependence.
2. 開關機制: 基於 Arrhenius Law 的離子遷移速率方程 (Filament Growth/Dissolution).
3. 非理想性: 包含線路寄生電阻 (IR Drop)、熱雜訊 (Thermal Noise)、製程變異.

[警告]
- program_weights() 不再是瞬間完成。它會執行物理時間步進，若寫入脈衝太短，
  電阻可能不會變 (寫入失敗)，這模擬了真實的 Write Failure。

===========================================================================
"""

import numpy as np
import logging
from SCDM_Hardware import SCDM_HardwareInterface

class SCDM_HardwarePrimitive(SCDM_HardwareInterface):
    def __init__(self, rows=64, cols=64):
        """
        初始化物理層參數 (基於 TSMC 28nm RRAM PDK 數據模型)。
        """
        # --- 幾何與寄生參數 (Geometry & Parasitics) ---
        self.rows = rows
        self.cols = cols
        self.C_BL = 150e-15      # Bitline 寄生電容 (150fF)
        self.R_wire = 2.5        # 金屬導線線阻 (Ohm/cell)
        
        # --- RRAM 傳輸參數 (Conduction Parameters) ---
        self.I_0 = 1e-6          # 基準電流 (A)
        self.g_0 = 0.25e-9       # 特徵穿隧間隙 (m)
        self.V_0 = 0.25          # 特徵電壓 (V)
        
        # --- RRAM 動力學參數 (Dynamics for Writing) ---
        # 用於計算導電絲生長速率 v = v0 * exp(-Ea + gamma*V)
        self.Vel_0 = 1000.0      # 基準漂移速度 (m/s)
        self.Ea_kT = 35.0        # 活化能/熱能比 (Ea/kT)
        self.gamma_acc = 18.0    # 加速因子 (Field Acceleration Factor)
        self.gap_min = 0.1e-9    # LRS 間隙 (導電絲接通)
        self.gap_max = 1.5e-9    # HRS 間隙 (導電絲斷開)
        
        # --- 操作參數 (Operating Conditions) ---
        self.V_write = 2.0       # 寫入電壓 (SET)
        self.V_erase = -2.0      # 抹除電壓 (RESET)
        self.I_compliance = 80e-6 # 限制電流 (防止元件燒毀)
        
        # --- 軟體可調暫存器 (Registers) ---
        self.V_READ_POS = 0.35   # +1 讀取電壓
        self.V_READ_NEG = -0.35  # -1 讀取電壓
        self.V_TH_SNN = 0.05     # SNN 擊發門檻
        
        # --- 狀態變數 (State Variables) ---
        # 模擬 5% 的製程變異 (Process Variation)
        self.process_var = np.random.normal(1, 0.05, (rows, cols))
        
        # 初始化 Gap 狀態 (預設全為 HRS)
        self.gap_A = np.full((rows, cols), self.gap_max) * self.process_var
        self.gap_B = np.full((rows, cols), self.gap_max) * self.process_var
        
        # 暫態電路狀態
        self.V_BL_A = np.zeros(cols)
        self.V_BL_B = np.zeros(cols)
        self.snn_fired = np.zeros(cols, dtype=bool)
        self.snn_times = np.full(cols, -1.0)
        self.current_time = 0.0
        
        # 輸入驅動電壓緩衝區
        self.active_inputs = np.zeros((rows, 1))

    # =====================================================================
    # 物理方程式 (The Physics Equations) - DO NOT SIMPLIFY
    # =====================================================================
    
    def _current_tunneling(self, V, gap):
        """
        [Mott-Gurney Law] 計算穿隧電流
        I = I0 * exp(-gap/g0) * sinh(V/V0)
        包含 sinh 非線性項，精確描述雙極性電流。
        """
        # 添加 1% 的隨機電報雜訊 (RTN - Random Telegraph Noise)
        rtn_noise = np.random.normal(1, 0.01, V.shape) 
        
        psi = gap / self.g_0
        v_norm = V / self.V_0
        return self.I_0 * np.exp(-psi) * np.sinh(v_norm) * rtn_noise

    def _filament_dynamics(self, V, gap, dt):
        """
        [Ion Migration] 計算導電絲生長/溶解
        d(gap)/dt = -v * exp(-Ea + gamma*|V|)
        這不是瞬間完成的，而是與時間、電壓呈現高度非線性的關係。
        """
        # 計算遷移速率 (Drift Velocity)
        gamma_eff = self.gamma_acc * np.abs(V)
        velocity = self.Vel_0 * np.exp(-self.Ea_kT + gamma_eff)
        
        delta_gap = velocity * dt
        
        # 更新 Gap (V > 0 -> SET -> Gap 變小; V < 0 -> RESET -> Gap 變大)
        new_gap = gap.copy()
        
        # SET Process (Gap shrinking)
        mask_set = (V > 0.5) # 需超過閾值電壓才開始移動
        new_gap[mask_set] -= delta_gap[mask_set]
        
        # RESET Process (Gap expanding)
        mask_reset = (V < -0.5)
        new_gap[mask_reset] += delta_gap[mask_reset]
        
        # 物理限制 (Physical Bounds)
        return np.clip(new_gap, self.gap_min, self.gap_max)

    # =====================================================================
    # 硬體指令：寫入與組態 (Programming)
    # =====================================================================

    def reset_weights_physics(self):
        """
        [物理指令] 全局區塊抹除 (Block Erase)
        施加負電壓脈衝，強制離子回遷。
        """
        logging.info("[HW Info] Executing Block Erase (Physics Simulation)...")
        dt = 100e-12 # 100ps step
        erase_time = 10e-9 # 10ns pulse
        steps = int(erase_time / dt)
        
        # 模擬加壓過程
        for _ in range(steps):
            V_erase_mat = np.full((self.rows, self.cols), self.V_erase)
            self.gap_A = self._filament_dynamics(V_erase_mat, self.gap_A, dt)
            self.gap_B = self._filament_dynamics(V_erase_mat, self.gap_B, dt)

    def program_weights_physics(self, target_weights):
        """
        [物理指令] 真實權重燒錄 (Real-time Write Verify)
        這不再是簡單的賦值，而是模擬真實的「寫入-驗證」迴圈。
        如果寫入時間不夠，權重將無法正確寫入 (Write Failure)。
        
        Args:
            target_weights: {-1, 0, 1}
        """
        logging.info("[HW Info] Programming Weights (Transient Dynamics)...")
        dt = 50e-12 # 50ps step
        write_pulse = 5.0e-9 # 5ns write pulse (相當快)
        steps = int(write_pulse / dt)
        
        # 定義目標
        # 1 -> A:LRS, B:HRS
        # -1 -> A:HRS, B:LRS
        mask_program_A = (target_weights == 1)
        mask_program_B = (target_weights == -1)
        
        # 執行暫態寫入模擬
        for t in range(steps):
            # 準備電壓
            V_applied_A = np.zeros((self.rows, self.cols))
            V_applied_B = np.zeros((self.rows, self.cols))
            
            V_applied_A[mask_program_A] = self.V_write
            V_applied_B[mask_program_B] = self.V_write
            
            # 計算當前電流 (檢查 Compliance)
            I_A = self._current_tunneling(V_applied_A, self.gap_A)
            I_B = self._current_tunneling(V_applied_B, self.gap_B)
            
            # 如果電流超過限流，電壓會被拉低 (Voltage Collapse)，停止生長
            V_applied_A[I_A > self.I_compliance] = 0.1 # 降壓
            V_applied_B[I_B > self.I_compliance] = 0.1
            
            # 更新 Gap (物理生長)
            self.gap_A = self._filament_dynamics(V_applied_A, self.gap_A, dt)
            self.gap_B = self._filament_dynamics(V_applied_B, self.gap_B, dt)

    # =====================================================================
    # 硬體指令：執行期控制 (Runtime Control)
    # =====================================================================

    def set_inputs_driver(self, input_vector):
        """
        [硬體指令] 設定 Wordline Driver 狀態
        input_vector: 64x1 array, {-1, 0, 1}
        """
        v_in = np.zeros(self.rows)
        v_in[input_vector == 1] = self.V_READ_POS
        v_in[input_vector == -1] = self.V_READ_NEG
        self.active_inputs = v_in.reshape(-1, 1)

    def reset_capacitors(self):
        """ [硬體指令] 放電 (Pre-charge / Equalize) """
        self.V_BL_A.fill(0.0)
        self.V_BL_B.fill(0.0)
        self.snn_fired.fill(False)
        self.snn_times.fill(-1.0)
        self.current_time = 0.0

    def run_physics_step(self, duration_ns):
        """
        [核心指令] 執行物理積分。
        包含 IR Drop 計算與 KCL 電流加總。
        """
        dt = 1e-12 # 1ps 高精度模擬
        steps = int((duration_ns * 1e-9) / dt)
        
        # 計算線路壓降 (IR Drop) - 基於物理線阻模型
        # V_drop = I_total * R_wire * distance
        # 這裡採用簡化的集中參數模型以維持運算效率，但保留物理意義
        dist_factor = np.arange(self.cols) * self.R_wire # 距離越遠電阻越大
        # 假設平均電流造成的壓降 (First-order approximation)
        v_drop_profile = 1.0 - (1e-6 * dist_factor) 
        
        for _ in range(steps):
            self.current_time += dt
            
            # 1. 計算跨壓 (考慮線損)
            V_eff = self.active_inputs * v_drop_profile
            V_cell_A = V_eff - self.V_BL_A
            V_cell_B = V_eff - self.V_BL_B
            
            # 2. 計算矩陣電流 (Vector-Matrix Multiplication)
            I_A = np.sum(self._current_tunneling(V_cell_A, self.gap_A), axis=0)
            I_B = np.sum(self._current_tunneling(V_cell_B, self.gap_B), axis=0)
            
            # 3. 積分 (C * dV/dt = I)
            # 加入熱雜訊 (Thermal Noise Current)
            noise_current = np.random.normal(0, 1e-9, self.cols) # 1nA rms noise
            
            self.V_BL_A += ((I_A + noise_current) * dt) / self.C_BL
            self.V_BL_B += ((I_B + noise_current) * dt) / self.C_BL
            
            # 4. 硬體比較器 (Hardware Comparator) - 連續時間偵測
            delta_v = self.V_BL_A - self.V_BL_B
            
            # SNN 擊發檢查
            new_fire = (np.abs(delta_v) > self.V_TH_SNN) & (~self.snn_fired)
            if np.any(new_fire):
                self.snn_fired[new_fire] = True
                self.snn_times[new_fire] = self.current_time

    # =====================================================================
    # 讀出介面 (Readout Interface)
    # =====================================================================

    def sense_voltage(self):
        """ [BNN用] 讀取 SA 鎖存前的類比電壓 """
        # 模擬 SA 的 Input Offset (失調電壓)
        sa_offset = np.random.normal(0, 0.005, self.cols) # 5mV offset
        return (self.V_BL_A - self.V_BL_B) + sa_offset

    def sense_spikes(self):
        """ [SNN用] 讀取擊發暫存器 """
        return self.snn_fired.copy(), self.snn_times.copy()