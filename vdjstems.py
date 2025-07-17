import torch
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model
import argparse
import soundfile as sf
import os
import ffmpeg
from pathlib import Path
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
from functools import partial
import time
from typing import List, Tuple, Optional
import psutil
from tqdm import tqdm
import threading

temp_dir = ".vdjstems_temp"

# Supported audio file extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.aiff', '.au', '.mp4'}

def find_audio_files(paths: List[str]) -> List[str]:
    """Find all audio files from given paths (files or directories)."""
    audio_files = []

    for path in paths:
        path_obj = Path(path)

        if path_obj.is_file():
            # Check if it's an audio file
            if path_obj.suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(str(path_obj))
            else:
                print(f"Skipping non-audio file: {path_obj.name}")
        elif path_obj.is_dir():
            # Recursively find audio files in directory
            found_files = []
            for ext in AUDIO_EXTENSIONS:
                found_files.extend(path_obj.rglob(f'*{ext}'))
                found_files.extend(path_obj.rglob(f'*{ext.upper()}'))

            # Convert to strings and add to list
            dir_audio_files = [str(f) for f in found_files]
            audio_files.extend(dir_audio_files)

            if dir_audio_files:
                print(f"Found {len(dir_audio_files)} audio files in {path}")
            else:
                print(f"No audio files found in {path}")
        else:
            print(f"Path not found: {path}")

    # Remove duplicates and sort
    audio_files = sorted(list(set(audio_files)))
    return audio_files

def create_vdjstems(audio_files: List[str], output_file: str, track_names: List[str]) -> None:
    # print(f"opt: {output_file}")
    """Create VDJ stems file from audio tracks."""
    # print(f"audio: {audio_files}")
    # print(f"output_file: {output_file}")
    # print(f"track_names: {track_names}")
    # Get duration from first audio file for empty track
    probe = ffmpeg.probe(audio_files[0], v='quiet')
    duration = float(probe['streams'][0]['duration'])

    # Process track names - rename "drums" to "kick" BEFORE creating inputs
    processed_track_names = []
    for name in track_names:
        # print(f"Processing track name: {name}")
        if name.lower() == "drums":
            processed_track_names.append("kick")
        else:
            processed_track_names.append(name)
    # print("3")
    # print(track)
    # Create input streams for existing audio files (in the same order as track_names)
    inputs = []
    for audio_file in audio_files:
        inputs.append(ffmpeg.input(audio_file))

    # Add empty hihat track
    silent_input = ffmpeg.input(
        'anullsrc=channel_layout=stereo:sample_rate=48000',
        f='lavfi',
        t=duration
    )
    inputs.append(silent_input)
    processed_track_names.append("hihat")

    # Build the output with multiple audio tracks
    output_kwargs = {
        'acodec': 'aac',
        'ar': 48000,  # Sample rate 48kHz
        'ab': '192k'  # Bitrate
    }

    # Create metadata for track names - this order must match the inputs order
    # print("1")
    # print(processed_track_names)
    metadata = {}
    for i, name in enumerate(processed_track_names):
        metadata[f'metadata:s:a:{i}'] = f'title={name}'
    # print("1")
    # print(metadata)
    # print(f"Track mapping: {list(zip(audio_files + ['silent_hihat'], processed_track_names))}")

    output = ffmpeg.output(
        *inputs,
        output_file,
        **output_kwargs,
        **metadata,
        f='matroska'
    )

    # Run the command with silenced output
    ffmpeg.run(output, overwrite_output=True, quiet=True)
    os.rename(output_file, f"{Path(output_file).stem}.vdjstems")

def load_audio_optimized(file_path: str, sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """Optimized audio loading with better memory management."""
    try:
        # Use librosa with optimized parameters
        waveform, sample_rate = librosa.load(
            file_path,
            sr=sr,
            mono=False,
            dtype=np.float32,  # Use float32 instead of float64
            res_type='kaiser_fast'  # Faster resampling
        )

        # Ensure correct shape for demucs (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform[None, :]  # Add channel dimension

        return torch.from_numpy(waveform).float(), sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def check_existing_output(file_path: str, output_dir: str) -> bool:
    """Check if a .vdjstems file already exists for the input file."""
    stem_name = Path(file_path).stem
    vdjstems_path = os.path.join(output_dir, f"{stem_name}.vdjstems")
    return os.path.exists(vdjstems_path)

def filter_files_to_process(files: List[str], output_dir: str) -> Tuple[List[str], List[str]]:
    """Filter out files that already have corresponding .vdjstems files."""
    files_to_process = []
    skipped_files = []

    for file_path in files:
        if check_existing_output(file_path, output_dir):
            skipped_files.append(file_path)
        else:
            files_to_process.append(file_path)

    return files_to_process, skipped_files

def write_stems_batch(stems_data: List[Tuple[np.ndarray, str, str]], sr: int) -> List[str]:
    """Write multiple stems in parallel using threading."""
    def write_single_stem(data: Tuple[np.ndarray, str, str]) -> str:
        audio_data, filename, temp_dir = data
        file_path = os.path.join(temp_dir, filename)
        sf.write(file_path, audio_data.T, sr)
        return file_path

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write_single_stem, data) for data in stems_data]
        return [future.result() for future in as_completed(futures)]
def process_single_file(args_tuple: Tuple[str, str, str, str, bool]) -> Optional[str]:
    """Process a single audio file - designed for multiprocessing."""
    file_path, output_dir, model_name, temp_base_dir, use_cuda = args_tuple

    # Create unique temp directory for this process
    process_id = os.getpid()
    process_temp_dir = f"{temp_base_dir}_{process_id}"

    try:
        # Create temp directory
        os.makedirs(process_temp_dir, exist_ok=True)

        # Load model (this will be done per process)
        model = get_model(model_name)
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        # Load audio with optimized settings
        waveform, sr = load_audio_optimized(file_path)

        # Apply model with batch processing
        with torch.no_grad():  # Disable gradient computation for inference
            sources = apply_model(
                model,
                waveform[None],
                device=device,
                progress=False,  # Disable progress bar in multiprocessing
                num_workers=1    # Use single worker per process
            )

        # Extract stems (order is important for demucs)
        drums, bass, other, vocals = sources[0]

        # Convert to numpy and define the order we want
        stems_numpy = [
            (vocals.cpu().numpy(), 'vocals.wav', 'vocal'),
            (drums.cpu().numpy(), 'drums.wav', 'drums'),
            (bass.cpu().numpy(), 'bass.wav', 'bass'),
            (other.cpu().numpy(), 'instrumental.wav', 'instruments')
        ]

        # Write stems and maintain order
        temp_files = []
        track_names = []

        # Process each stem in the defined order
        for audio_data, filename, track_name in stems_numpy:
            stem_file_path = os.path.join(process_temp_dir, filename)
            sf.write(stem_file_path, audio_data.T, sr)
            temp_files.append(stem_file_path)
            track_names.append(track_name)

        # Create output file (use original input file name)
        output_file = os.path.join(output_dir, f"{Path(file_path).stem}.mkv")
        create_vdjstems(temp_files, output_file, track_names)

        return output_file

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    finally:
        # Clean up temp directory
        if os.path.exists(process_temp_dir):
            shutil.rmtree(process_temp_dir)

def get_optimal_worker_count() -> int:
    """Determine optimal number of workers based on system resources."""
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    memory_gb = psutil.virtual_memory().total / (1024**3)

    # Estimate memory usage per worker (roughly 2-4GB per model)
    estimated_memory_per_worker = 3  # GB
    max_workers_by_memory = max(1, int(memory_gb / estimated_memory_per_worker))

    # Don't exceed CPU cores or memory constraints
    optimal_workers = min(cpu_count, max_workers_by_memory, 4)  # Cap at 4 for stability

    print(f"System: {cpu_count} CPU cores, {memory_gb:.1f}GB RAM")
    print(f"Using {optimal_workers} workers")
    return optimal_workers

def main():
    parser = argparse.ArgumentParser(prog='vdjstems')
    parser.add_argument('-d', "--debug", help="print out some useful info", action="store_true")
    parser.add_argument('-o', "--output", help="output path", required=False, default="./output")
    parser.add_argument('-m', "--model", help="model name", type=str,
                       choices="htdemucs htdemucs_ft htdemucs_6s hdemucs_mmi mdx mdx_extra mdx_q mdx_extra_q SIG".split(" "),
                       default="htdemucs")
    parser.add_argument('-j', "--jobs", help="number of parallel jobs", type=int, default=0)
    parser.add_argument('--no-cuda', help="disable CUDA", action="store_true")
    parser.add_argument('--force', help="force processing even if output exists", action="store_true")
    parser.add_argument('-r', "--recursive", help="recursively search directories for audio files", action="store_true")
    parser.add_argument('paths', nargs='*', help='audio files or directories to process')

    args = parser.parse_args()

    if args.debug:
        librosa.show_versions()
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"CPU count: {mp.cpu_count()}")
        print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        print(f"Supported audio extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        return

    if not args.paths:
        print("No files or directories provided")
        return

    # Find all audio files from the provided paths
    print("Scanning for audio files...")
    audio_files = find_audio_files(args.paths)

    if not audio_files:
        print("No audio files found!")
        return

    print(f"Found {len(audio_files)} audio files")

    # Show some examples of found files
    if len(audio_files) <= 10:
        print("Files to process:")
        for f in audio_files:
            print(f"  - {Path(f).name}")
    else:
        print("Sample of files to process:")
        for f in audio_files[:5]:
            print(f"  - {Path(f).name}")
        print(f"  ... and {len(audio_files) - 5} more")
    print()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Filter files to process (skip existing unless --force)
    if args.force:
        files_to_process = audio_files
        skipped_files = []
    else:
        files_to_process, skipped_files = filter_files_to_process(audio_files, args.output)

    # Show skip information
    if skipped_files:
        print(f"Skipping {len(skipped_files)} files (already processed):")
        for file_path in skipped_files:
            print(f"  - {Path(file_path).name}")
        print()

    if not files_to_process:
        print("No files to process!")
        return

    # Determine number of workers
    if args.jobs == 0:
        num_workers = get_optimal_worker_count()
    else:
        num_workers = args.jobs

    use_cuda = not args.no_cuda

    print(f"Processing {len(files_to_process)} files with {num_workers} workers")
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print()

    # Create progress bar
    progress_bar = tqdm(
        total=len(files_to_process),
        desc="Processing files",
        unit="file",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    # Prepare arguments for multiprocessing (removed progress_bar)
    process_args = [
        (file_path, args.output, args.model, temp_dir, use_cuda)
        for file_path in files_to_process
    ]

    # Process files
    successful_files = []
    failed_files = []

    if num_workers == 1:
        # Single-threaded processing
        for arg_tuple in process_args:
            result = process_single_file(arg_tuple)
            if result:
                successful_files.append(result)
            else:
                failed_files.append(arg_tuple[0])
            progress_bar.update(1)
    else:
        # Multi-process processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_file, args): args[0] for args in process_args}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        successful_files.append(result)
                    else:
                        failed_files.append(futures[future])
                except Exception as e:
                    print(f"Worker failed processing {futures[future]}: {e}")
                    failed_files.append(futures[future])

                progress_bar.update(1)

    progress_bar.close()

    # Final summary
    print("\nProcessing completed!")
    print(f"Successfully processed: {len(successful_files)} files")
    if failed_files:
        print(f"Failed to process: {len(failed_files)} files")
    if skipped_files:
        print(f"Skipped (already exist): {len(skipped_files)} files")

    print(f"Total files handled: {len(successful_files) + len(failed_files) + len(skipped_files)}")

    # List successful outputs
    if successful_files:
        print(f"\nOutput files created in {args.output}:")
        for output_file in successful_files:
            vdjstems_file = f"{Path(output_file).stem}.vdjstems"
            print(f"  - {vdjstems_file}")

    print("\nAll processing completed!")

if __name__ == "__main__":
    main()
