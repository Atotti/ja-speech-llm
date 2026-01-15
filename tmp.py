from demo2_ja import LlamaForSpeechLM, LlamaForSpeechLMConfig

model = LlamaForSpeechLM(LlamaForSpeechLMConfig(
    encoder_id="openai/whisper-large-v3",
    decoder_id="/groups/gch51701/Team031/model/pretrained/v4-8b-decay2m-ipt_v3.1-instruct4"
)).cuda()

print("=== Before unfreeze ===")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Unfreeze decoder
model.decoder.requires_grad_(True)

print("\n=== After unfreeze ===")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# Check decoder specifically
decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
print(f"Decoder trainable: {decoder_trainable:,}")
