import numpy as np

class Processor:
    def __init__(self, rate: int=48000):
        self.rate=rate

    def process_audio(self, mic: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
        """
        Process the audio from the microphone and reference signal.

        Parameters:
        mic (np.ndarray): The microphone signal.
        ref (np.ndarray): The reference signal.

        Returns:
        np.ndarray: The processed audio signal.
        """
        #Passthrough filter
        return mic