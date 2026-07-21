import pandas as pd
import openai
import time
import os
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# OpenAI APIキーの設定
def setup_openai():
    """OpenAIクライアントの設定"""
    openai.api_key = os.getenv("OPENAI_API_KEY")
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
    - Phrase A (Assumption): "have_evident_staff_not_speak_English_properly"
    - Phrase B (Proposition): "helpful_staff"
    - Answer: False (not contrary - not-English-speaker staff does not mean helpful staff)

    Example 1:
    - Phrase A (Assumption): "have_evident_staff_should_learn_better_English"
    - Phrase B (Proposition): "easy_to_communicate_with_staff"
    - Answer: True (contrary - If staff should learn better English, they are not easy to communicate with)
    
    Example 2:
    - Phrase A (Assumption): "have_evident_staff_need_improve_collaboration"
    - Phrase B (Proposition): "best_staff_ever"
    - Answer: True (contrary - If staff need to improve collaboration, they are not the best staff ever)

    Example 3:
    - Phrase A (Assumption): "have_evident_no_empathy_staff"
    - Phrase B (Proposition): "good_hospitality_staff"
    - Answer: True (contrary - No empathy staff and good hospitality staff are contradictory)

    Example 4:
    - Phrase A (Assumption): "have_evident_no_empathy_staff"
    - Phrase B (Proposition): "ready_to_help_guest"
    - Answer: False (not contrary - It is possible that no empathy staff is ready to help guest)
        
    Now analyze this pair:
    - Phrase A (Assumption): "{assumption}"
    - Phrase B (Proposition): "{proposition}"
    
    Question: Is phrase B a contrary of phrase A?
    
    Please answer with only "True" or "False".
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a logical reasoning expert. Answer only with 'True' or 'False'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
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

def process_csv_with_gpt(input_file_path, output_file_path, sample_size=None, start_from=0, end_row=None):
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
    
    # 出力ファイルが既に存在する場合、既存のisContraryカラムの値を読み込む
    if os.path.exists(output_file_path):
        print(f"既存の出力ファイル {output_file_path} を検出しました。")
        existing_df = pd.read_csv(output_file_path)
        
        # 既存のisContraryカラムがある場合、その値をコピー
        if 'isContrary' in existing_df.columns:
            df['isContrary'] = existing_df['isContrary'].fillna("")
            
            # 値が入っている最後の行を見つける
            last_filled_idx = -1
            for idx in range(len(df)):
                if pd.notna(df.at[idx, 'isContrary']) and df.at[idx, 'isContrary'] != "":
                    last_filled_idx = idx
            
            # start_fromを自動調整（既に処理済みの行はスキップ）
            if last_filled_idx >= 0:
                auto_start = last_filled_idx + 1
                if auto_start > start_from:
                    print(f"既に {last_filled_idx + 1} 行目まで処理済みです。")
                    print(f"処理を {auto_start + 1} 行目（インデックス: {auto_start}）から開始します。")
                    start_from = auto_start
        else:
            df['isContrary'] = ""
    else:
        df['isContrary'] = ""
    
    # 処理範囲を決定
    total_rows = len(df)
    if end_row is None: 
        end_row = total_rows

    # if sample_size:
    #     end_row = min(start_from + sample_size, end_row)
    
    # 全て処理済みの場合
    if start_from >= total_rows:
        print(f"全ての行が既に処理済みです（総行数: {total_rows}）")
        return df
    
    print(f"行 {start_from + 1} から {end_row} まで処理します (全 {end_row - start_from} 件)")
    
    print("GPT-4o-miniへの問い合わせを開始...")
    
    # start_fromから処理を開始
    rows_to_process = df.iloc[start_from:end_row]
    processed_count = 0
    
    for i, (index, row) in enumerate(tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Processing")):
        # 既に値が入っている行はスキップ
        if pd.notna(df.at[index, 'isContrary']) and df.at[index, 'isContrary'] != "":
            continue
            
        assumption = row['Assumption']
        proposition = row['Proposition']
        
        # GPT-4o-miniに問い合わせ
        result = ask_gpt_contrary(assumption, proposition)
        df.at[index, 'isContrary'] = result
        processed_count += 1
        
        # API制限を避けるために少し待機
        time.sleep(0.5)
        
        # 進捗を定期的に保存（50件ごと）
        if processed_count > 0 and processed_count % 50 == 0:
            df.to_csv(output_file_path, index=False)
            print(f"進捗保存: {processed_count} 件処理完了")
    
    # 最終結果を保存
    df.to_csv(output_file_path, index=False)
    print(f"処理完了！{processed_count} 件の新規処理を行い、結果を {output_file_path} に保存しました")
    
    return df

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='GPT-4o-miniを使用してContrary判定を行うツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # 基本的な使用方法
  python gpt_4omini_contMakeAnno.py -i input.csv -o output.csv
  
  # 特定の範囲を処理
  python gpt_4omini_contMakeAnno.py -i input.csv -o output.csv --start 100 --end 500
  
  # テストモード（10件のみ処理）
  python gpt_4omini_contMakeAnno.py -i input.csv -o output.csv --test
        '''
    )
    
    # 必須引数
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='入力CSVファイルのパス'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='出力CSVファイルのパス'
    )
    
    # オプション引数
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='処理開始行番号（0ベース、デフォルト: 0）'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='処理終了行番号（指定しない場合は最後まで処理）'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='テストモード（10件のみ処理）'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='処理するサンプル数（指定しない場合は全て処理）'
    )
    
    parser.add_argument(
        '--no-prompt',
        action='store_true',
        help='確認プロンプトをスキップ'
    )
    
    return parser.parse_args()

def main():
    """メイン実行関数"""
    # コマンドライン引数を解析
    args = parse_arguments()
    
    input_file = args.input
    output_file = args.output
    start_from = args.start
    end_row = args.end
    
    print("=== GPT-4o-mini Contrary Analysis ===")
    print(f"入力ファイル: {input_file}")
    print(f"出力ファイル: {output_file}")
    print(f"開始行: {start_from + 1} 行目")
    
    if end_row is not None:
        print(f"終了行: {end_row} 行目")
    else:
        print("終了行: ファイルの最後まで")
    
    # テストモードまたはサンプルサイズの処理
    if args.test:
        sample_size = 10
        print("テストモード：10件のサンプルで実行します")
        if end_row is None:
            end_row = start_from + 10
    elif args.sample_size:
        sample_size = args.sample_size
        print(f"サンプルモード：{sample_size}件を処理します")
        if end_row is None:
            end_row = start_from + sample_size
    else:
        sample_size = None
        if not args.no_prompt:
            # 確認プロンプト
            confirm = input("全データを処理します。続行しますか？ (y/n): ").lower()
            if confirm != 'y':
                print("処理を中止しました")
                return
    
    try:
        # ファイルの存在確認
        if not os.path.exists(input_file):
            print(f"エラー: 入力ファイル '{input_file}' が見つかりません")
            return
        
        # 出力ディレクトリの作成
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"出力ディレクトリを作成: {output_dir}")
        
        result_df = process_csv_with_gpt(input_file, output_file, sample_size, start_from, end_row)
        
        # 結果の要約を表示
        print("\n=== 結果要約 ===")
        print(f"処理済み件数: {len(result_df)}")
        
        # 処理済みの範囲内での集計
        processed_range = result_df.iloc[start_from:end_row if end_row else len(result_df)]
        print(f"True の件数: {(processed_range['isContrary'] == 'True').sum()}")
        print(f"False の件数: {(processed_range['isContrary'] == 'False').sum()}")
        print(f"未処理の件数: {(processed_range['isContrary'] == '').sum()}")
        
        # Trueの割合を表示
        total_processed = (processed_range['isContrary'] != '').sum()
        if total_processed > 0:
            true_ratio = (processed_range['isContrary'] == 'True').sum() / total_processed * 100
            print(f"Contrary (True) の割合: {true_ratio:.1f}%")
        
    except FileNotFoundError:
        print(f"エラー: ファイル {input_file} が見つかりません")
    except KeyError as e:
        print(f"エラー: CSVファイルに必要なカラムがありません: {e}")
        print("必要なカラム: 'Assumption', 'Proposition'")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 