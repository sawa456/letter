# 訓練データとして使用する画像を読み込み、適切な形式に前処理します。
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 散らかった部屋と整頓された部屋の画像を格納したディレクトリ
dir_messy = r"C:\Users\masa3\Downloads\kikagaku_app\app\DL\messy"
dir_clean = r"C:\Users\masa3\Downloads\kikagaku_app\app\DL\clean"

# 画像データとラベルデータを格納するリスト
images = []
labels = []

# 散らかった部屋の画像を読み込み、ラベルと共にリストに追加
for file in os.listdir(dir_messy):
    image = cv2.imread(os.path.join(dir_messy, file))
    if image is None:
        print(f"画像の読み込みに失敗しました: {os.path.join(dir_messy, file)}")
        continue
    image = cv2.resize(image, (128, 128)) / 255.0  # ピクセル値をスケール
    images.append(image)
    labels.append(0)  # 散らかった部屋を0とラベル付け

# 整頓された部屋の画像を読み込み、ラベルと共にリストに追加
for file in os.listdir(dir_clean):
    image = cv2.imread(os.path.join(dir_clean, file))
    if image is None:
        print(f"画像の読み込みに失敗しました: {os.path.join(dir_clean, file)}")
        continue
    image = cv2.resize(image, (128, 128)) / 255.0   # ピクセル値をスケール
    images.append(image)
    labels.append(1)  # 整頓された部屋を1とラベル付け




# リストをNumPy配列に変換
images = np.array(images)
labels = np.array(labels)

# データセットを訓練用とテスト用に分割
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)


# Kerasを使って画像分類モデルを設定します。
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# モデルの定義
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())  # バッチ正規化層を追加

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())  # バッチ正規化層を追加

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # ドロップアウト層を追加
model.add(Dense(1, activation='sigmoid'))  

# コンパイル
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# データ拡張
datagen = ImageDataGenerator(
        rotation_range=20,  # 画像をランダムに回転する回転範囲
        zoom_range = 0.2, # ランダムにズームする範囲
        width_shift_range=0.2,  # ランダムに水平シフトする範囲
        height_shift_range=0.2,  # ランダムに垂直シフトする範囲
        horizontal_flip=True,  # ランダムに画像を水平反転
        vertical_flip=True)  # ランダムに画像を垂直反転

# データ拡張をモデルの訓練に適用
datagen.fit(x_train)

# 訓練
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          epochs=10, validation_data=(x_test, y_test))

# モデルの保存
model.save("room_model02.h5")

# ファイルが存在することをチェックするコード
import os
if os.path.isfile("room_model02.h5"):
    print("モデルは正常に保存されました。")
else:
    print("モデルの保存に失敗しました。")

