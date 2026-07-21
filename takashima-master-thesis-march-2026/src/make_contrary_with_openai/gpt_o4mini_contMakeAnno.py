import pandas as pd
import openai
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# OpenAI APIキーの設定
def setup_openai():
    """OpenAIクライアントの設定"""
    openai.api_key = os.getenv("API_KEY")
    return openai

def ask_gpt_contrary(assumption, proposition):
    """
    GPT-4o-miniに対してContrary判定を問い合わせる関数
    
    Args:
        assumption (str): 仮定文
        proposition (str): 命題文
        client: OpenAIクライアント
    
    Returns:
        str: "True" または "False"
    """
    
    prompt = f"""
    You are an expert in logical reasoning and hotel review analysis.
    
    Your task is to determine if phrase B is a contrary of phrase A in the context of hotel reviews.
    A contrary relationship means that the two phrases express opposite (direct negation) or contradictory concepts.
    
    Here are some examples:

    Example 0:
    - Phrase A (Assumption): "no_evident_not_clean_room"
    - Phrase B (Proposition): "dirty_room"
    - Answer: True (contrary - direct negation)

    Example 1:
    - Phrase A (Assumption): "no_evident_not_clean_room"
    - Phrase B (Proposition): "loud_air_conditioner"
    - Answer: False (not contrary - different aspects)
    
    Example 2:
    - Phrase A (Assumption): "no_evident_not_comfortable_bed"
    - Phrase B (Proposition): "too_hard_mattress"
    - Answer: True (contrary - "too hard" is a negative aspect of a comfortable bed and both are unable to exist at the same time)
    
    Example 3:
    - Phrase A (Assumption): "no_evident_not_comfortable_bed"
    - Phrase B (Proposition): "small_room"
    - Answer: False (not contrary - small room and comfortable bed are possible to exist at the same time)
    
    Example 4:
    - Phrase A (Assumption): "no_evident_not_nice_window_view"
    - Phrase B (Proposition): "too_close_to_the_airport"
    - Answer: False (not contrary - Phrase B could mean negative aspects of Phrase A for some people who don't like airport view, but not the direct negation)
    
    Example 5:
    - Phrase A (Assumption): "no_evident_not_clean_room"
    - Phrase B (Proposition): "no_bodysoap_in_bathroom"
    - Answer: False (contrary - Although phrase B might cause having a dirty body, phrase B does not cause the direct negation of phrase A.)
    
    Example 6:
    - Phrase A (Assumption): "no_evident_not_clean_room"
    - Phrase B (Proposition): "need_better_cleaner"
    - Answer: True (contrary - needing better cleaning contradicts no evidence of unclean room)

    Example 7:
    - Phrase A (Assumption): "no_evident_not_clean_room"
    - Phrase B (Proposition): "stinky_bathroom"
    - Answer: True  (contrary - stinky means not cleaning well and it is a direct negation of phrase A)
    
    Now analyze this pair:
    - Phrase A (Assumption): "{assumption}"
    - Phrase B (Proposition): "{proposition}"
    
    Question: Is phrase B a contrary of phrase A?
    
    Please answer with only "True" or "False".
    """
    
    try:
        response = openai.chat.completions.create(
            model="o4-mini-2025-04-16",
            messages=[
                {"role": "system", "content": "You are a logical reasoning expert. Answer only with 'True' or 'False'."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        if content is None:
            print("GPTからのレスポンスが空でした")
            return "False"
            
        answer = content.strip()
        # 答えが"True"か"False"でない場合のフォールバック
        if answer.lower() == "true":
            return "True"
        elif answer.lower() == "false":
            return "False"
        else:
            print(f"Unexpected response: {answer}")
            return "False"  # デフォルトはFalse
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "False"  # エラー時はFalse

def process_csv_with_gpt(input_file_path, output_file_path, sample_size=None, start_from=0):
    """
    CSVファイルを読み込んでGPT-4o-miniでContrary判定を行い、結果を保存
    
    Args:
        input_file_path (str): 入力CSVファイルのパス
        output_file_path (str): 出力CSVファイルのパス
        sample_size (int, optional): 処理するサンプル数（Noneの場合は全て処理）
        start_from (int): 開始行番号（0ベース）
    """
    
    print("CSVファイルを読み込み中...")
    df = pd.read_csv(input_file_path)
    df['isContrary'] = ""
    
    # 処理範囲を決定
    total_rows = len(df)
    start_from = 14550
    end_row = total_rows

    if sample_size:
        end_row = min(start_from + sample_size, total_rows)
    
    print(f"行 {start_from} から {end_row-1} まで処理します (全 {end_row - start_from} 件)")
    
    print("GPT-4o-miniへの問い合わせを開始...")
    
    # start_fromから処理を開始
    rows_to_process = df.iloc[start_from:end_row]
    for i, (index, row) in enumerate(tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Processing")):
        assumption = row['Assumption']
        proposition = row['Proposition']
        
        # GPT-4o-miniに問い合わせ
        result = ask_gpt_contrary(assumption, proposition)
        df.at[index, 'isContrary'] = result
        
        # API制限を避けるために少し待機
        time.sleep(0.5)
        
        # 進捗を定期的に保存（50件ごと）
        if (i + 1) % 50 == 0:
            df.to_csv(output_file_path, index=False)
            print(f"進捗保存: {start_from + i + 1} 件完了")
    
    # 最終結果を保存
    df.to_csv(output_file_path, index=False)
    print(f"処理完了！結果を {output_file_path} に保存しました")
    
    return df

def main():
    """メイン実行関数"""
    input_file = "data/RoomSilver_ContP_BodyN.csv"
    output_file = "data/RoomSilver_Pos2Neg_with_isCont_o4mini.csv"
    
    print("=== GPT-4o-mini Contrary Analysis ===")
    print(f"入力ファイル: {input_file}")
    print(f"出力ファイル: {output_file}")
    
    # 開始行の設定
    start_from = 2350  # 14550行目から開始（0ベースなので-1）
    print(f"開始行: {start_from + 1} 行目")
    
    # テストモードの選択
    test_mode = input("テストモードで実行しますか？ (y/n): ").lower() == 'y'
    
    if test_mode:
        sample_size = 10
        print("テストモード：10件のサンプルで実行します")
    else:
        sample_size = None
        print("指定行から最後まで全データを処理します")
    
    try:
        result_df = process_csv_with_gpt(input_file, output_file, sample_size, start_from)
        
        # 結果の要約を表示
        print("\n=== 結果要約 ===")
        print(f"処理済み件数: {len(result_df)}")
        print(f"True の件数: {(result_df['isContrary'] == 'True').sum()}")
        print(f"False の件数: {(result_df['isContrary'] == 'False').sum()}")
        print(f"未処理の件数: {(result_df['isContrary'] == '').sum()}")
        
    except FileNotFoundError:
        print(f"エラー: ファイル {input_file} が見つかりません")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main() 