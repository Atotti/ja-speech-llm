import torch

# SYSTEM_PROMPT と build_multiturn_prompt をインポート
import sys
sys.path.append('/home/ayu/GitHub/ja-speech-llm')
from gradio_demo import SYSTEM_PROMPT, build_multiturn_prompt

# モックのconversation_turnsを作成
def test_text_audio_text_flow():
    """テキスト→音声→テキストのフローをテスト"""

    conversation_turns = []

    # ターン1：テキスト入力（キャラクター設定）
    turn1_instruction = "あなたは「こはる」という名前のAIアシスタントです。元気で素直な性格の女の子です。"
    turn1_response = "はい、こはるです！よろしくお願いします♪"
    conversation_turns.append({
        "instruction": turn1_instruction,
        "response": turn1_response,
        # audio_featuresなし（テキストターン）
    })

    # ターン2：音声入力
    turn2_instruction = "音声の指示に従ってください。"
    turn2_response = "こんにちは！元気にしてますか？"
    turn2_audio_features = torch.randn(128, 100)  # モックの音声特徴
    conversation_turns.append({
        "instruction": turn2_instruction,
        "response": turn2_response,
        "audio_features": turn2_audio_features,
    })

    # ターン3：テキスト入力（現在のターン）
    turn3_instruction = "あなたの名前は何ですか？"

    # プロンプトを構築
    prompt = build_multiturn_prompt(conversation_turns, turn3_instruction, current_has_audio=False)

    # 結果を表示
    print("=" * 80)
    print("テスト：テキスト→音声→テキストのフロー")
    print("=" * 80)
    print(f"\nconversation_turns count: {len(conversation_turns)}")
    for i, turn in enumerate(conversation_turns):
        has_audio = "audio_features" in turn
        print(f"  Turn {i+1}: has_audio={has_audio}, instruction='{turn['instruction'][:50]}...'")

    print(f"\n現在のターン instruction: '{turn3_instruction}'")
    print(f"\n構築されたプロンプト:\n{'-'*80}")
    print(prompt)
    print('-'*80)

    # 検証
    assert turn1_instruction in prompt, "❌ ターン1のテキスト指示がプロンプトに含まれていない！"
    assert turn2_instruction in prompt, "❌ ターン2の指示がプロンプトに含まれていない！"
    assert turn3_instruction in prompt, "❌ ターン3の指示がプロンプトに含まれていない！"
    assert turn1_response in prompt, "❌ ターン1の応答がプロンプトに含まれていない！"
    assert turn2_response in prompt, "❌ ターン2の応答がプロンプトに含まれていない！"

    print("\n✅ すべてのチェックに合格！")

if __name__ == "__main__":
    test_text_audio_text_flow()
