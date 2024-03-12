"""
zip_unziper.py : ZIPファイルを指定されたディレクトリに解凍するスクリプト
"""

import zipfile

def unzip_file(zip_path, extract_to):
    """
    ZIPファイルを指定されたディレクトリに解凍する。

    Parameters:
    zip_path (str): 解凍するZIPファイルのパス
    extract_to (str): ファイルを解凍する先のディレクトリパス
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"{zip_path}は{extract_to}に正常に保存されました。")

if __name__ == "__main__":
    zip_path = "./data/penn-fudan.zip"
    extract_to = "./data/"
    unzip_file(zip_path, extract_to)

