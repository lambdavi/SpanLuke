import json
import os
def convert_format(all_data, compute_doc=False):
    entitities = ['B-COURT', 'B-PETITIONER', 'B-RESPONDENT', 'B-JUDGE', 'B-DATE', 'B-ORG', 'B-GPE', 'B-STATUTE', 'B-PROVISION', 'B-PRECEDENT', 'B-CASE_NUMBER', 'B-WITNESS', 'B-OTHER_PERSON', 'B-LAWYER', 'I-COURT', 'I-PETITIONER', 'I-RESPONDENT', 'I-JUDGE', 'I-DATE', 'I-ORG', 'I-GPE', 'I-STATUTE', 'I-PROVISION', 'I-PRECEDENT', 'I-CASE_NUMBER', 'I-WITNESS', 'I-OTHER_PERSON', 'I-LAWYER']
    entity_to_tag = {e: i+1 for i, e in enumerate(sorted(entitities))}
    entity_to_tag["O"]=0
    all_output_data=[]   
    doc_split = dict()
    f_index = 0 
    for input_data in all_data:
        # Extract text and annotations from the input data
        text = input_data["data"]["text"]
        annotations = input_data["annotations"][0]["result"]

        # Initialize lists for tokens and NER tags
        tokens = []
        ner_tags = [0] * len(text.split())

        doc_word = input_data["meta"]["source"].split(" ")[1]
        if doc_split.get(doc_word, None) is None:
            doc_split[doc_word] = [0, len(doc_split.keys())]
        else:
            doc_split[doc_word][0]+=1

        # Populate tokens and NER tags based on annotations
        for annotation in annotations:
            entity = annotation["value"]["labels"][0]
            words = annotation["value"]["text"].split()
            ner_tag=['B-'+entity]
            if len(words)!=1:
                ner_tag+=(['I-'+entity]*(len(words)-1))
            
            final_tags = [entity_to_tag[n] for n in ner_tag]
            start = annotation["value"]["start"]
            end = annotation["value"]["end"]
                
            # Update NER tags for the corresponding token positions
            start_token = len(text[:start].split())
            end_token = start_token + len(text[start:end].split())
            ner_tags[start_token:end_token] = final_tags

        # Populate tokens list
        tokens = text.split()

        # Create the final output dictionary
        if compute_doc:
            all_output_data.append({"tokens": tokens, "ner_tags": ner_tags, "document_id": doc_split[doc_word][1], "sentence_id": doc_split[doc_word][0]})
        else:
            all_output_data.append({"tokens": tokens, "ner_tags": ner_tags})
        
    return all_output_data



# Example usage for the first entry
splits = ["TRAIN", "DEV"]
print(os.getcwd())
for split in splits:
    input_data = json.load(open(f"data/NER_{split}/NER_{split}_ALL.json"))
    output_data = convert_format(input_data, compute_doc=True)
    with open(f"data/NER_{split}/NER_{split}_ALL_DOC3.jsonl", "w") as f:
        for line in output_data:
            f.write(json.dumps(line)+'\n')