import os
from model_export import ModelImportHandler

from model_executor import ModelExecutor
from RRAM_Driver import RRAM_Simulator
from model_loading import TernaryBitNetLoader


def main():
    print("🚀 Filament Competitive RRAM Simulator 啟動流程開始...")

    # 請將這裡替換成你真實的 BitNet 模型權重路徑
    MODEL_PATH = ModelImportHandler.MODEL_DEFAULT_DIR / "NLP_BitNet_2B4T_TNN.pickle"
    TernaryBitNetLoader.MAX_TOKEN = 20
    TernaryBitNetLoader.FIX_OUTPUT_GENERATION = True
    SIMULATION_TF = True

    # ==========================================
    # 1. create Executor
    # ==========================================
    executor = ModelExecutor()

    # ==========================================
    # 2. create Driver
    # ==========================================
    driver = RRAM_Simulator()

    # ==========================================
    # 3. put driver into Executor
    # ==========================================
    if SIMULATION_TF == True:
        executor.config_driver(driver)

    # ==========================================
    # 4. choose Model Loader
    # ==========================================
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ [警告] 找不到模型路徑: {MODEL_PATH}，若繼續執行可能會在載入時報錯。")
    loader = TernaryBitNetLoader(MODEL_PATH)

    # ==========================================
    # 5. load Model Loader into Executor
    # ==========================================
    executor.load(loader)

    # ==========================================
    # [第一回合推論測試]
    # ==========================================
    print("\n" + "="*40)
    print("🎯 第一回合測試開始")
    print("="*40)

    # 6. prepare data
    # prompt_1 = "Hello!"
    prompt_1 = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    input_data_1 = TernaryBitNetLoader.str2token(prompt_1)

    # 7. run executor
    # 呼叫 Stream 模式產生文字
    stream_result_1 = executor.run(input_data_1, input_command=TernaryBitNetLoader.STREAM_GENERATE_COMMAND)
    
    print(f"\nPrompt: {prompt_1}")
    print("🤖 模型輸出: ", end="", flush=True)
    TernaryBitNetLoader.print_stream(stream_result_1)
    print("\n")

    # 8. log statistic data
    executor.save_info()

    # ==========================================
    # [第二回合推論測試]
    # ==========================================
    print("\n" + "="*40)
    print("🎯 第二回合測試開始")
    print("="*40)

    # 9. go back to 6 or move forward
    prompt_2 = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Can you explain what quantum computing is?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    input_data_2 = TernaryBitNetLoader.str2token(prompt_2)

    # 選擇性：清除上一回合累積的 Ops 統計，讓第二回合從零開始算
    executor.clear_info_cache()

    # 再次執行 Step 7
    stream_result_2 = executor.run(input_data_2, input_command=TernaryBitNetLoader.STREAM_GENERATE_COMMAND)
    
    print("\n🤖 模型輸出: ", end="")
    TernaryBitNetLoader.print_stream(stream_result_2)
    print("\n")

    # 再次執行 Step 8
    executor.save_info()

    # ==========================================
    # 10. unload Model Loader
    # ==========================================
    executor.unload()

    # ==========================================
    # 11. go back to 4 or move forward
    # ==========================================
    print("\nWorkflow completed. 模擬結束！")

if __name__ == "__main__":
    main()
