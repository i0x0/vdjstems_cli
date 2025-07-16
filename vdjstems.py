from importlib.resources import files
import torch
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model
import argparse
from string import printable

p = argparse.ArgumentParser(prog='vdjstems')
p.add_argument('-d', "--debug", help="print out some useful info", action="store_true")
p.add_argument('-o', "--output", help="output path", required=False)
p.add_argument('-m', "--model", help="model name", type=str, choices="htdemucs htdemucs_ft htdemucs_6s hdemucs_mmi mdx mdx_extra mdx_q mdx_extra_q SIG".split(" "), default="htdemucs")
p.add_argument('files', nargs='*')
args = p.parse_args()
if args.debug:
    librosa.show_versions()
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    p.exit()

model = get_model(args.model)
print(args.files)
for file in args.files:
    waveform, sr = librosa.load(file, sr=44100, mono=False)
    print(1)
    waveform = torch.tensor(waveform).float()
    print(2)
    sources = apply_model(model, waveform[None], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(3)

    print(sources)
