from SCDM_Driver import SCDM_DriverInterface

import numpy

class BasicModelLoader():
    """
    ModelLoader
    ===============================================
    1. 介於 已訓練好的model 與 ModelExecutor
    2. 將 parsing model (每種 model 一個)
    3. 將 model 拆解成一步一步的
    4. 定義哪幾步可以由 SCDM 運算
    
    ===============================================
    """

    PREDICT_COMMAND = "predict_command"

    def config(self, driver: SCDM_DriverInterface):
        """
        配置硬體
        TODO
        """
        self.driver = driver

    def run(self, input_data: numpy.ndarray, input_command: str = None) -> numpy.ndarray:
        """
        吃 input_data
        根據 input_command 決定執行方法
        TODO
        """
        pass

    def unload(self):
        """
        把模型卸載
        例如把 weight 從硬體中刪掉
        TODO
        """
        pass
