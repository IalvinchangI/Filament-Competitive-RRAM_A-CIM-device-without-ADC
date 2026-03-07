"""
2026/2/5 
Author : 陳昱軒
Description: This is physic compact model for our device
"""
import numpy as np
import matplotlib.pyplot as plt

class Y_RRAM_Final_Engineer_Version:
    def __init__(self, rows=256, cols=256):
        self.rows = rows
        self.cols = cols
        
        # ==========================================
        # 1. 動力學參數 (Filament Kinetics)
        # ==========================================
        self.v_0 = 100.0         # 嘗試頻率因子
        self.E_a = 0.9 * 1.6e-19 # 活化能 0.9 eV
        self.k_B = 1.38e-23      # Boltzmann constant
        
        # 溫度設定 
        self.T_room = 300.0      # Read Temp
        self.T_write = 800.0     # Write Temp (High for kinetics)
        
        self.gamma_k = 4      # Field acceleration factor
        self.gap_min_phy = 0.2e-9 
        self.gap_max_phy = 2.5e-9 
        
        # Y-Shape 耦合係數 (Vacancy Stealing)
        self.coupling_factor = 0.5 
        
        # 寫入操作參數
        self.V_WRITE = 2.5       
        self.dt_write = 10e-12   
        
        # ==========================================
        # 2. 讀取與電路參數 (Read Physics)
        # ==========================================
        self.V_READ = 0.35
        self.I_0 = 10e-6 
        self.gamma_IV = 0.30e-9
        
        # 寄生參數 (256x256 Array)
        self.C_int = 55e-15      # Integration Cap
        self.C_par_wl = 100e-15  # WL Parasitic (Gate+Wire)
        self.C_par_bl = 100e-15  # BL Parasitic (Drain+Wire)
        
        # OCSA Parameters
        self.V_TH_SA = 0.015     # Threshold Voltage
        
        # PPA Parameters (Unit: Joule)
        self.E_SA = 0.05e-12     # 50fJ per SA op
        self.E_Digital = 0.03e-12 # Shift-Adder energy
        
        # 初始化權重與 Gap (High-Z Reset State)
        self.weights = np.zeros((rows, cols))
        self.gaps_A = np.full((rows, cols), self.gap_max_phy)
        self.gaps_B = np.full((rows, cols), self.gap_max_phy)
        
        self.total_energy = 0.0

    def _update_coupled_kinetics(self, V_A, V_B, g_A, g_B, dt, Temp):
        """
        [物理核心] Y-Shape 耦合動力學
        A 和 B 的生長是互斥的 (Competitive Growth)
        """
        thermal = np.exp(-self.E_a / (self.k_B * Temp))
        
        # 注意: rate 為負代表 Gap 變小 (生長)
        rate_A = -self.v_0 * thermal * np.sinh(self.gamma_k * V_A) * dt
        rate_B = -self.v_0 * thermal * np.sinh(self.gamma_k * V_B) * dt
        
        # 引入耦合 (Coupling / Stealing)
        coupling_from_B = self.coupling_factor * np.minimum(0, rate_B)
        coupling_from_A = self.coupling_factor * np.minimum(0, rate_A)
        
        delta_g_A = rate_A - coupling_from_B
        delta_g_B = rate_B - coupling_from_A
        
        new_g_A = np.clip(g_A + delta_g_A, self.gap_min_phy, self.gap_max_phy)
        new_g_B = np.clip(g_B + delta_g_B, self.gap_min_phy, self.gap_max_phy)
        
        return new_g_A, new_g_B

    def program_weights_physically(self, target_weights):
        """
        模擬真實寫入 (含 Node Voltage 定義)
        """
        print(f">>> [Step 1] Programming Weights on {self.rows}x{self.cols} Array (Coupled Kinetics)...")
        steps = int(10e-9 / self.dt_write) # 10ns pulse
        
        V_TE = np.zeros_like(target_weights, dtype=float)
        V_BEA = np.zeros_like(target_weights, dtype=float)
        V_BEB = np.zeros_like(target_weights, dtype=float)
        
        # +1: Set A
        mask_p1 = (target_weights == 1)
        V_TE[mask_p1] = self.V_WRITE
        V_BEA[mask_p1] = 0.0
        V_BEB[mask_p1] = self.V_WRITE 
        
        # -1: Set B
        mask_n1 = (target_weights == -1)
        V_TE[mask_n1] = self.V_WRITE
        V_BEA[mask_n1] = self.V_WRITE 
        V_BEB[mask_n1] = 0.0
        
        # 0: Reset Both
        mask_0 = (target_weights == 0)
        V_TE[mask_0] = 0.0
        V_BEA[mask_0] = self.V_WRITE 
        V_BEB[mask_0] = self.V_WRITE 
        
        for _ in range(steps):
            DeltaV_A = V_TE - V_BEA
            DeltaV_B = V_TE - V_BEB
            self.gaps_A, self.gaps_B = self._update_coupled_kinetics(
                DeltaV_A, DeltaV_B, self.gaps_A, self.gaps_B, self.dt_write, self.T_write
            )
            
        lrs_idx = (target_weights == 1)
        if np.any(lrs_idx):
            avg_gA = np.mean(self.gaps_A[lrs_idx])
            avg_gB = np.mean(self.gaps_B[lrs_idx])
            print(f"    >>> Verification (+1 State):")
            print(f"        Gap A (Selected):   {avg_gA*1e9:.2f} nm (Target ~0.2nm)")
            print(f"        Gap B (Unselected): {avg_gB*1e9:.2f} nm (Target ~2.5nm)")

    def _read_current(self, V, g):
        g_eff = np.maximum(g, 0.1e-9)
        return self.I_0 * np.exp(-g_eff / self.gamma_IV) * np.sinh(V / 0.25)

    def run_inference_full_stack(self, input_8bit):
        P_STATIC_LEAKAGE = 50e-6  
        E_CLOCK_PER_COL = 10e-15  
        E_ZERO_CHECK = 0.5e-12    
        REAL_E_DIGITAL = 0.08e-12 
        
        accumulator = np.zeros(self.cols, dtype=int)
        t_inference_total = 8 * 400e-12
        self.total_energy += P_STATIC_LEAKAGE * t_inference_total
        steps_read = int(200e-12 / 10e-12)
        
        for bit in range(8):
            self.total_energy += self.cols * E_CLOCK_PER_COL
            self.total_energy += E_ZERO_CHECK
            wl_vec = (input_8bit >> bit) & 1
            
            if np.sum(wl_vec) > 0:
                active_rows = np.sum(wl_vec)
                e_ac = (active_rows * self.C_par_wl + self.cols * self.C_par_bl) * (self.V_READ**2)
                self.total_energy += e_ac
                
                V_drv = wl_vec.reshape(-1, 1) * self.V_READ
                V_cA = np.zeros(self.cols)
                V_cB = np.zeros(self.cols)
                
                for _ in range(steps_read):
                    V_cell_A = np.maximum(V_drv - V_cA, 0)
                    V_cell_B = np.maximum(V_drv - V_cB, 0)
                    
                    I_A = self._read_current(V_cell_A, self.gaps_A)
                    I_B = self._read_current(V_cell_B, self.gaps_B)
                    
                    self.total_energy += np.sum(V_cell_A*I_A + V_cell_B*I_B) * 10e-12
                    
                    noise = np.random.normal(0, 5e-9, self.cols)
                    V_cA += (np.sum(I_A, axis=0) + noise) * 10e-12 / self.C_int
                    V_cB += (np.sum(I_B, axis=0) + noise) * 10e-12 / self.C_int
                    
                    V_cA = np.clip(V_cA, 0, 0.35)
                    V_cB = np.clip(V_cB, 0, 0.35)
                
                V_sig = V_cA - V_cB + np.random.normal(0, 0.005, self.cols)
                latched_out = np.zeros(self.cols, dtype=int)
                latched_out[V_sig > self.V_TH_SA] = 1
                latched_out[V_sig < -self.V_TH_SA] = -1
                
                self.total_energy += self.cols * self.E_SA
                shift_val = latched_out * (1 << bit)
                accumulator += shift_val
                self.total_energy += self.cols * REAL_E_DIGITAL
                
        return accumulator

# =========================================================================
# Main Execution Flow (All tests strictly on 256x256)
# =========================================================================
ARRAY_DIM = 256 # 絕對不簡化的 256x256 規格
print(f"=== 啟動全棧驗證 (Full-Stack Verification) - 陣列規格: {ARRAY_DIM}x{ARRAY_DIM} ===")

# 1. 引擎初始化
engine = Y_RRAM_Final_Engineer_Version(ARRAY_DIM, ARRAY_DIM)
target_W = np.random.choice([-1, 0, 1], size=(ARRAY_DIM, ARRAY_DIM), p=[0.2, 0.6, 0.2])
engine.program_weights_physically(target_W)

# -------------------------------------------------------------------------
# [Test 1] Sparsity vs Efficiency Sweep
# -------------------------------------------------------------------------
print("\n>>> [Test 1] 系統稀疏度與能效掃描 (Sparsity Sweep)...")
sparsity_levels = np.linspace(0.0, 0.99, 10) # 減少 sweep 點數以節省時間，但保留 256x256 精度
tops_w_results = []
ops_per_inference = 2 * ARRAY_DIM * ARRAY_DIM 

for sp in sparsity_levels:
    engine.total_energy = 0.0
    X_in = np.zeros(ARRAY_DIM, dtype=int)
    num_non_zeros = int(ARRAY_DIM * (1 - sp))
    if num_non_zeros > 0:
        indices = np.random.choice(ARRAY_DIM, num_non_zeros, replace=False)
        X_in[indices] = np.random.randint(1, 128, size=num_non_zeros)
    
    _ = engine.run_inference_full_stack(X_in)
    
    if engine.total_energy > 1e-20:
        tops_w = (ops_per_inference / 1e12) / engine.total_energy
    else:
        tops_w = 0
    tops_w_results.append(tops_w)
    print(f"    Sparsity: {sp*100:5.1f}% | Efficiency: {tops_w:8.2f} TOPS/W")

# -------------------------------------------------------------------------
# [Test 2] Accuracy & Linearity Check
# -------------------------------------------------------------------------
print("\n>>> [Test 2] 準確率與線性度檢測 (Linearity Check)...")
hw_results_pool, math_results_pool = [], []
for i in range(100): # 測試 30 根 256 維度的向量
    X_test = np.zeros(ARRAY_DIM, dtype=int)
    mask = np.random.rand(ARRAY_DIM) > 0.5 
    X_test[mask] = np.random.randint(1, 128, size=np.sum(mask))
    
    hw_out = engine.run_inference_full_stack(X_test)
    math_out = np.dot(X_test, target_W)
    
    hw_results_pool.extend(hw_out)
    math_results_pool.extend(math_out)

hw_arr = np.array(hw_results_pool)
math_arr = np.array(math_results_pool)
correlation = np.corrcoef(math_arr, hw_arr)[0, 1]
print(f"    Pearson Correlation: {correlation:.4f} (Target > 0.90)")

# -------------------------------------------------------------------------
# [Test 3] 寫入電壓邊界測試 (V_WRITE Margin Analysis)
# -------------------------------------------------------------------------
print(f"\n>>> [Test 3] 寫入電壓邊界掃描 ({ARRAY_DIM}x{ARRAY_DIM} 動力學引擎)...")
v_write_sweep = np.linspace(1.0, 3.5, 60)
gap_A_results = []
gap_B_results = []

for vw in v_write_sweep:
    engine.V_WRITE = vw
    engine.gaps_A = np.full((ARRAY_DIM, ARRAY_DIM), engine.gap_max_phy)
    engine.gaps_B = np.full((ARRAY_DIM, ARRAY_DIM), engine.gap_max_phy)
    
    steps = int(10e-9 / engine.dt_write)
    V_TE = np.zeros_like(target_W, dtype=float)
    V_BEA = np.zeros_like(target_W, dtype=float)
    V_BEB = np.zeros_like(target_W, dtype=float)
    mask_p1 = (target_W == 1)
    V_TE[mask_p1] = engine.V_WRITE
    V_BEA[mask_p1] = 0.0
    V_BEB[mask_p1] = engine.V_WRITE 
    
    for _ in range(steps):
        DeltaV_A = V_TE - V_BEA
        DeltaV_B = V_TE - V_BEB
        engine.gaps_A, engine.gaps_B = engine._update_coupled_kinetics(
            DeltaV_A, DeltaV_B, engine.gaps_A, engine.gaps_B, engine.dt_write, engine.T_write
        )
    
    gap_A_results.append(np.mean(engine.gaps_A[mask_p1]) * 1e9)
    gap_B_results.append(np.mean(engine.gaps_B[mask_p1]) * 1e9)
    print(f"    V_WRITE = {vw:.1f}V -> Gap A: {gap_A_results[-1]:.2f}nm, Gap B: {gap_B_results[-1]:.2f}nm")

# -------------------------------------------------------------------------
# [Test 4] 讀取電壓折衷分析 (V_READ vs. Energy-Linearity Pareto)
# -------------------------------------------------------------------------
print("\n>>> [Test 4] 讀取電壓折衷分析 (V_READ Pareto)...")
# 恢復正常寫入電壓並執行標準寫入
engine.V_WRITE = 2.5
engine.gaps_A = np.full((ARRAY_DIM, ARRAY_DIM), engine.gap_max_phy)
engine.gaps_B = np.full((ARRAY_DIM, ARRAY_DIM), engine.gap_max_phy)
engine.program_weights_physically(target_W)

v_read_sweep = np.linspace(0.1, 0.6, 25)
energy_per_inf = []
linearity_scores = []

for vr in v_read_sweep:
    engine.V_READ = vr
    engine.total_energy = 0.0
    hw_res_vr, math_res_vr = [], []
    
    for _ in range(10): # 每個電壓測試 10 個 256-D 向量
        X_test = np.random.randint(0, 128, ARRAY_DIM)
        hw_out = engine.run_inference_full_stack(X_test)
        math_out = np.dot(X_test, target_W)
        hw_res_vr.extend(hw_out)
        math_res_vr.extend(math_out)
        
    std_hw = np.std(hw_res_vr)
    corr = np.corrcoef(math_res_vr, hw_res_vr)[0, 1] if std_hw > 1e-6 else 0.0
    linearity_scores.append(corr)
    energy_per_inf.append((engine.total_energy * 1e12) / 10) 
    print(f"    V_READ = {vr:.2f}V -> Linearity: {corr:.4f}, Energy: {energy_per_inf[-1]:.1f} pJ")

# -------------------------------------------------------------------------
# [Test 5] 製程變異抗性測試 (Process Variation & Reliability)
# -------------------------------------------------------------------------
print("\n>>> [Test 5] 類比變異抗性測試 (Process Variation vs Reliability)...")
sigma_sweep = np.linspace(0.0, 0.5, 40) 
variation_corrs = []

# 保存完美的原始 Gap
golden_gap_A = np.copy(engine.gaps_A)
golden_gap_B = np.copy(engine.gaps_B)
engine.V_READ = 0.35 # 恢復最佳讀取電壓

for sigma in sigma_sweep:
    # 注入物理層變異
    noise_A = np.random.normal(0, sigma, engine.gaps_A.shape)
    noise_B = np.random.normal(0, sigma, engine.gaps_B.shape)
    
    engine.gaps_A = np.clip(golden_gap_A * (1 + noise_A), engine.gap_min_phy, engine.gap_max_phy)
    engine.gaps_B = np.clip(golden_gap_B * (1 + noise_B), engine.gap_min_phy, engine.gap_max_phy)
    
    hw_res_var, math_res_var = [], []
    for _ in range(15): # 測試 15 根向量
        X_test = np.random.randint(0, 128, ARRAY_DIM)
        hw_out = engine.run_inference_full_stack(X_test)
        math_out = np.dot(X_test, target_W)
        hw_res_var.extend(hw_out)
        math_res_var.extend(math_out)
        
    std_hw = np.std(hw_res_var)
    corr = np.corrcoef(math_res_var, hw_res_var)[0, 1] if std_hw > 1e-6 else 0.0
    variation_corrs.append(corr)
    print(f"    Gap Variation = {sigma*100:4.1f}% -> Correlation: {corr:.4f}")

# 恢復完美 Gap
engine.gaps_A = golden_gap_A
engine.gaps_B = golden_gap_B

# =========================================================================
# 繪製所有結果圖表
# =========================================================================
print("\n>>> 正在生成獨立評估分析圖表...")

# --- Plot 1: Linearity Scatter (Test 2) ---
plt.figure(figsize=(8, 6))
plt.scatter(math_arr, hw_arr, alpha=0.3, color='purple', s=10)
m, b = np.polyfit(math_arr, hw_arr, 1)
plt.plot(math_arr, m*math_arr + b, color='red', linestyle='--', linewidth=2)
plt.title(f'Math(REAL) vs. Hardware (Corr={correlation:.4f})', fontsize=14)
plt.xlabel('Ideal INT16 Result')
plt.ylabel('Hardware ADC-less Output')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: V_WRITE Kinetics (Test 3) ---
plt.figure(figsize=(8, 6))
plt.plot(v_write_sweep, gap_A_results, 'ro-', linewidth=2, label='Selected (BEA)')
plt.plot(v_write_sweep, gap_B_results, 'go-', linewidth=2, label='Inhibit (BEB)')
plt.axhline(y=0.2, color='k', linestyle='--', label='Limit')
plt.title('Kinetics Gap vs. Write Voltage', fontsize=14)
plt.xlabel('V_WRITE (V)')
plt.ylabel('Filament Gap (nm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 3: V_READ Pareto (Test 4) ---
fig, ax3 = plt.subplots(figsize=(8, 6))
ax3_2 = ax3.twinx()
lns1 = ax3.plot(v_read_sweep, energy_per_inf, 'b^-', linewidth=2, label='Energy')
lns2 = ax3_2.plot(v_read_sweep, linearity_scores, 'ms-', linewidth=2, label='Linearity')

# 合併圖例 (因為有雙 Y 軸)
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc=0)

ax3.set_title('V_READ Trade-off (Energy vs Linearity)', fontsize=14)
ax3.set_xlabel('V_READ (V)')
ax3.set_ylabel('Energy (pJ)', color='b')
ax3_2.set_ylabel('Correlation', color='m')
ax3.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 4: Variation Immunity (Test 5) ---
plt.figure(figsize=(8, 6))
plt.plot(sigma_sweep * 100, variation_corrs, 'co-', linewidth=2)
plt.axhline(y=0.85, color='r', linestyle='--', label='Threshold (0.85)')
plt.title('Reliability vs. Physical Mismatch', fontsize=14)
plt.xlabel('Physical Gap Variation (%)')
plt.ylabel('Linearity (Correlation)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
