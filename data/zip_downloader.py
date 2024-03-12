"""
zip_downloader.py : 指定されたURLからZIPファイルをダウンロードして保存するスクリプト
"""

import requests
import os

def download_zip(url, save_path):
    """
    指定されたURLからZIPファイルをダウンロードして保存する。

    Parameters:
    url (str): ZIPファイルのダウンロードURL。
    save_path (str): 保存するファイルのパス(ファイル名を含む)。
    """
    response = requests.get(url)
    # レスポンスが200、つまりOKの場合、ファイルを保存する
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"ファイルが{save_path}に正常に保存されました。")
    else:
        print(f"ダウンロードに失敗しました。ステータスコード：{response.status_code}")

if __name__ == "__main__":
    # 使用例
    url = "https://www.cis.upenn.edu/%7Ejshi/ped_html/PennFudanPed.zip"
    save_path = "./data/penn-fudan.zip"
    if not os.path.exists("./data"):
        os.makedirs("data")
    download_zip(url, save_path)
