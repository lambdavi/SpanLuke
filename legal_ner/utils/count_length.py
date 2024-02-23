import json
import numpy as np

print("** INFORMATON ABOUT LEGAL_NER **")
for split in ["TRAIN", "DEV"]:
    all_data = json.load(open(f"data/NER_{split}/NER_{split}_ALL.json"))
    ent_l = [] 
    for input_data in all_data:
        # Extract text and annotations from the input data
        text = input_data["data"]["text"]
        annotations = input_data["annotations"][0]["result"]
       
        # Populate tokens and NER tags based on annotations
        for annotation in annotations:
            words = annotation["value"]["text"].split()
            ent_l.append(len(words))
            if len(words)==43:
                print(words)

    print(f"{split.upper()} SUMMARY:")
    med = int(len(ent_l)/2)
    print(f"Tot number of entities: {len(ent_l)}")
    print(f"Avg. Number of words per entity: {np.mean(ent_l)}")
    print(f"Std. Number of words per entity: {np.std(ent_l)}")
    print(f"Mode Number of words per entity: {sorted(ent_l)[med]}")
    print(f"Max Number of words per entity: {np.max(ent_l)}")
    print(f"Min Number of words per entity: {np.min(ent_l)}")
    print("\n\n")
    ent_l = []



