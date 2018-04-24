import nnfc_codec


ctx = nnfc_codec.EncoderContext()
print(ctx.encode("hello world"))

print(nnfc_codec.available_encoders())
print(nnfc_codec.available_decoders())
