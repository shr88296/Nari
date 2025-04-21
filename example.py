import soundfile as sf

from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B")

with open("example.txt") as f:
    text = f.read()

output = model.generate(
    text=text,
    cfg_scale=4.0,
    temperature=1.2,
    top_p=0.95,
    use_cfg_filter=True,
    cfg_filter_top_k=50,
    use_torch_compile=True,
)

sf.write("example.mp3", output, 44100)
