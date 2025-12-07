from stable_baselines3.common.callbacks import BaseCallback
import os

class CheckpointCallback(BaseCallback):
    """
    학습 중간 과정을 저장하고 영상을 기록하기 위한 콜백.
    일정 간격(save_freq)마다 모델을 저장합니다.
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}_steps")
            self.model.save(model_path)
            if self.verbose > 1:
                print(f"Sasving model checkpoint to {model_path}")
        return True
