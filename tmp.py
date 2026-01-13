from datasets import load_dataset

ds = load_dataset('Atotti/spoken-magpie-ja', split='train', streaming=True)
print('=== Keys and sample ===')
sample = next(iter(ds))
print('Keys:', list(sample.keys()))

# Check for duration metadata
for k in sample.keys():
    if 'duration' in k.lower() or 'length' in k.lower():
        print(f'Found duration/length column: {k} = {sample[k]}')

# Check audio duration from a few samples
print('\n=== Audio durations (first 10 samples) ===')
ds = load_dataset('Atotti/spoken-magpie-ja', split='train', streaming=True)
for i, sample in enumerate(ds):
    if i >= 10:
        break
    audio = sample['instruction_audio']
    duration = len(audio['array']) / audio['sampling_rate']
    print(f'Sample {i}: {duration:.1f}s')
