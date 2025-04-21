from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B")

output = model.generate(
    text="[S1] Hello, world!",
    max_tokens=1000,
    cfg_scale=4.0,
    temperature=1.2,
    top_p=0.95,
)

model.save_audio(output, "example.mp3")
