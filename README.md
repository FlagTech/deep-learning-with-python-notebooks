# 《Deep Learning with Python》第三版：旗標版範例 Notebook

本倉庫提供 François Chollet 與 Matthew Watson 合著的 [《Deep Learning with Python》第三版（2025）](https://www.manning.com/books/deep-learning-with-python-third-edition?a_aid=keras&a_bid=76564dff)範例程式，供旗標繁體中文版讀者搭配本書使用。範例以 Jupyter Notebook 格式提供，並保留[第二版（2021）](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff)與[第一版（2017）](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff)的 Notebook，分別放在 second_edition 與 first_edition 目錄。

為方便閱讀，Notebook 只保留可執行的程式碼與小節標題，不包含書中的說明文字、圖片與虛擬碼。**建議搭配本書閱讀，才能掌握各段程式的用途與原理。**

## 旗標修正版說明

本書第 14～17 章所使用的原版 Notebook 中，部分範例存在函式庫版本、參數設定、資料集網址、程式碼錯字等問題，另有部分內容無法在 T4 GPU 環境下順利執行。

因此，我們另外提供檔名加上 _flag 的修正版 Notebook，並會在書中各章開頭說明相關修改內容與注意事項。下方目錄的第 14～17 章連結指向旗標修正版；原版 Notebook 仍保留在本倉庫中，方便對照。

第 14 章提供兩種修正版：

- **T4 示範版（_flag-t4）**：縮小資料量、序列長度與詞彙表，並減少訓練週期，供 Colab 免費 T4 環境使用。訓練設定與結果會與書中不同，請勿將其視為書中結果的重現。
- **完整負載版（_flag-full）**：保留原版的資料量與訓練設定，修正相容性問題，請使用 A100 40 GB GPU 環境。

修正版不表示所有範例都能在 T4 上執行。例如，第 16 章的 Gemma 4B 範例仍需使用 A100 40 GB GPU；執行前請先閱讀書中及 Notebook 的相關注意事項。

## 執行程式碼

建議使用 [Google Colab](https://colab.google) 執行 Notebook。Colab 提供雲端執行環境，請依各份 Notebook 開頭的指示安裝所需套件。你也可以自行建立 Jupyter 環境，在本機執行，或參考 Colab 的[本機執行階段設定說明](https://research.google.com/colaboratory/local-runtimes.html)。

多數範例可使用 Colab 免費方案的 GPU 執行，但部分範例需要更多 GPU 記憶體，請依各章說明選擇適合的硬體。若使用 Colab Pro 等付費方案，第 8～18 章的範例可受益於較快的 GPU。你可以在 Colab 的「執行階段 → 變更執行階段類型」中選擇硬體加速器；實際可用的 GPU 依 Colab 分配情況而定。

## 選擇後端

第三版的程式使用 Keras 3，可選擇 JAX、TensorFlow 或 PyTorch 作為後端。若要設定後端，請修改 Notebook 開頭的下列程式碼：

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
```

每次工作階段只需設定一次，而且必須在匯入 Keras 之前完成。若已開始執行 Notebook，請透過「執行階段 → 重新啟動工作階段」重新啟動，再執行相關儲存格。

旗標修正版可能因相容性需求指定特定後端，請優先依照該份 Notebook 的設定與說明執行。

## 使用 Kaggle 資料

本書使用 Kaggle 提供的資料集與模型權重。Kaggle 是線上機器學習社群與平台；執行相關範例前，需要先建立 Kaggle 帳號，操作說明見第 8 章。

需要 Kaggle 資料的章節，可在執行到 kagglehub.login() 的儲存格時登入，每次工作階段登入一次。也可以將登入資訊存入 Colab 的「密鑰」：

1. 前往 [Kaggle](https://www.kaggle.com/) 登入。
2. 前往[帳號設定](https://www.kaggle.com/settings)，產生 Kaggle API 金鑰。
3. 在 Colab 左側點選鑰匙圖示，開啟「密鑰」面板。
4. 新增 KAGGLE_USERNAME 與 KAGGLE_KEY 兩個密鑰，分別填入使用者名稱與 API 金鑰。

設定後不必每次重新複製金鑰，但執行相關程式時，仍需允許各份 Notebook 存取這些密鑰。

## Notebook 目錄

以下連結可直接在 Colab 開啟本倉庫的 Notebook。

- [第 2 章：神經網路的數學基礎](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter02_mathematical-building-blocks.ipynb)
- [第 3 章：TensorFlow、PyTorch、JAX 與 Keras 簡介](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter03_introduction-to-ml-frameworks.ipynb)
- [第 4 章：開始使用神經網路：分類與迴歸問題](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter04_classification-and-regression.ipynb)
- [第 5 章：機器學習的基礎](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter05_fundamentals-of-ml.ipynb)
- [第 7 章：深入探討 Keras](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter07_deep-dive-keras.ipynb)
- [第 8 章：影像分類](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter08_image-classification.ipynb)
- [第 9 章：現代卷積神經網路架構模式](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter09_convnet-architecture-patterns.ipynb)
- [第 10 章：詮釋卷積神經網路學到的東西](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter10_interpreting-what-convnets-learn.ipynb)
- [第 11 章：圖像分割](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter11_image-segmentation.ipynb)
- [第 12 章：物體偵測](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter12_object-detection.ipynb)
- [第 13 章：時間序列預測](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter13_timeseries-forecasting.ipynb)
- 第 14 章：文字分類（旗標修正版）
  - [T4 示範版](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter14_text-classification_flag-t4.ipynb)
  - [完整負載版（A100 40 GB）](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter14_text-classification_flag-full.ipynb)
- [第 15 章：語言模型與 Transformer（旗標修正版）](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter15_language-models-and-the-transformer_flag.ipynb)
- [第 16 章：文字生成（旗標修正版）](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter16_text-generation_flag.ipynb)
- [第 17 章：影像生成（旗標修正版）](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter17_image-generation_flag.ipynb)
- [第 18 章：實務上的最佳實踐](https://colab.research.google.com/github/FlagTech/deep-learning-with-python-notebooks/blob/master/chapter18_best-practices-for-the-real-world.ipynb)
