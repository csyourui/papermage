from argparse import ArgumentParser
import json

import torch
import tqdm
from necessary import necessary
from sklearn.metrics import classification_report

from papermage.magelib import Document, Image, Entity, Span, Box, Metadata
from papermage.predictors import HFBIOTaggerPredictor, IVILATokenClassificationPredictor

with necessary("datasets"):
    import datasets


def from_json(cls, entity_json: dict) -> "Entity":
    # the .get(..., None) or [] pattern is to handle the case where the key is present but the value is None
    return cls(
        spans=[Span.from_json(span_json=span_json) for span_json in entity_json.get("spans", None) or []],
        boxes=[Box.from_json(box_json=box_json) for box_json in entity_json.get("boxes", None) or []],
        metadata=Metadata.from_json(entity_json.get("metadata", None) or {}),
    )


Entity.from_json = classmethod(from_json)   # type: ignore


ap = ArgumentParser()
ap.add_argument("vila", choices=["new", "old", "grobid"])
args = ap.parse_args()

dt = datasets.load_dataset('allenai/s2-vl', split='test')
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.vila == "new":
    vila_predictor = HFBIOTaggerPredictor.from_pretrained(
        model_name_or_path="allenai/vila-roberta-large-s2vl-internal",
        entity_name="tokens",
        context_name="pages",
        device=device,
    )
elif args.vila == "old":
    vila_predictor = IVILATokenClassificationPredictor.from_pretrained(
        "allenai/ivila-row-layoutlm-finetuned-s2vl-v2", device=device
    )
elif args.vila == "grobid":
    vila_predictor = None
else:
    raise ValueError(f"Invalid value for `vila`: {args.vila}")

docs = []

gold_tokens = []
pred_tokens = []

for row in tqdm.tqdm(dt, desc="Predicting", unit="doc"):
    doc = Document.from_json(row["doc"])
    images = [Image.from_base64(image) for image in row["images"]]
    doc.annotate_images(images=images)
    docs.append(doc)

    if args.vila != "grobid":
        entities = vila_predictor.predict(doc=doc)
        doc.annotate_entity(entities=entities, field_name="vila_entities")

        gold_tokens.extend(e[0].metadata.type if len(e := token._vila_entities) else "null" for token in doc.tokens)
        pred_tokens.extend(e[0].metadata.label if len(e := token.vila_entities) else "null" for token in doc.tokens)

    else:
        path = f"/net/nfs2.s2-research/aps/papermage/dataset/{doc.metadata.sha}-{doc.metadata.page:02d}.json"
        with open(path, "r") as f:
            grobid_doc = Document.from_json(json.load(f))
        breakpoint()



print(classification_report(y_true=gold_tokens, y_pred=pred_tokens, digits=4))