import gdown
import os

if __name__ == "__main__":
    if not os.path.exists('saved/model'):
        os.mkdir('saved/model')
    gdown.download("https://drive.google.com/file/d/13V_IdQI_JCwzngyqAq616E4ckjcFRg7E/view?usp=sharing", "saved/model/best_model.pth")
