from biotransformers import BioTransformers

sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]

bio_trans = BioTransformers(backend="protbert")
embeddings = bio_trans.compute_embeddings(
    sequences, pool_mode=("cls", "mean"), batch_size=2
)

cls_emb = embeddings["cls"]
mean_emb = embeddings["mean"]
print(cls_emb[0].shape)
print(mean_emb[0].shape)