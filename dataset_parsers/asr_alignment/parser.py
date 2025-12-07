# whisper/dataset_parsers/asr_alignment/parser.py
import os
import torchaudio
from datasets import load_dataset
import torch

def load_asr_alignment_split(subset: str, split: str = "train[:100]"):
    """
    Load a subset of the ASR-Alignment dataset with word-level alignments.

    Args:
        subset: One of the dataset subsets, e.g., "commonvoice", "gigaspeech", "tedlium".
        split: Hugging Face-style split, e.g., "train[:100]" for a small test slice.

    Returns:
        List of dicts, each dict containing:
            - 'audio': torch.Tensor waveform
            - 'words': list of dicts with 'start', 'end', 'word'
    """
    print(f"Loading ASR-Alignment subset '{subset}' split '{split}'...")
    ds = load_dataset("nguyenvulebinh/asr-alignment", subset, split=split)
    data = []
    for i, item in enumerate(ds):
        waveform = torch.tensor(item["audio"]["array"])
        # zip words with their start/end times
        words = [
            {"start": s, "end": e, "word": w}
            for w, s, e in zip(item["words"], item["word_start"], item["word_end"])
        ]
        data.append({"audio": waveform, "words": words})
        if i < 3:
            print(f"Sample {i}: audio shape {waveform.shape}, words {len(words)}")
            for w in words[:5]:
                print(f"  {w}")
    return data

def create_next_word_examples(dataset):
    """
    Convert ASR-Alignment dataset into next-word prediction examples.

    Args:
        dataset: Output of load_asr_alignment_split

    Returns:
        List of dicts:
            - 'audio_input': torch.Tensor waveform up to the next word
            - 'next_word': str
            - 'start': float (word start time)
            - 'end': float (word end time)
    """
    examples = []
    for sample in dataset:
        audio = sample["audio"]
        for w in sample["words"]:
            end_idx = int(w["start"] * 16000)  # stop BEFORE the next word starts
            audio_segment = audio[:end_idx]    # slice from start of sample to word start
            examples.append({
                "audio_input": audio_segment,
                "next_word": w["word"],
                "start": w["start"],
                "end": w["end"]
            })
    print(f"Created {len(examples)} next-word examples")
    return examples

def save_audio_examples(examples, n=5, folder="temp_audio_examples"):
    """
    Save first n audio examples to disk for playback.

    Args:
        examples: List of next-word dicts
        n: Number of examples to save
        folder: Directory to save audio files
    """
    os.makedirs(folder, exist_ok=True)
    for i, ex in enumerate(examples[:n]):
        filename = os.path.join(folder, f"example_{i}.wav")
        torchaudio.save(filename, ex["audio_input"].unsqueeze(0), 16000)
        print(f"Saved {filename}, next word: {ex['next_word']}")
