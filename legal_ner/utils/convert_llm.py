import json

def convert_format(all_data):
    entitities = ['B-COURT', 'B-PETITIONER', 'B-RESPONDENT', 'B-JUDGE', 'B-DATE', 'B-ORG', 'B-GPE', 'B-STATUTE', 'B-PROVISION', 'B-PRECEDENT', 'B-CASE_NUMBER', 'B-WITNESS', 'B-OTHER_PERSON', 'B-LAWYER', 'I-COURT', 'I-PETITIONER', 'I-RESPONDENT', 'I-JUDGE', 'I-DATE', 'I-ORG', 'I-GPE', 'I-STATUTE', 'I-PROVISION', 'I-PRECEDENT', 'I-CASE_NUMBER', 'I-WITNESS', 'I-OTHER_PERSON', 'I-LAWYER']
    entity_to_tag = {e: i+1 for i, e in enumerate(sorted(entitities))}
    entity_to_tag["O"]=0
    all_output_data=[]    
    for input_data in all_data:
        # Extract text and annotations from the input data
        text = input_data["data"]["text"]
        annotations = input_data["annotations"][0]["result"]

        # Initialize lists for tokens and NER tags
        tokens = []
        ner_tags = [0] * len(text.split())
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
        all_output_data.append({"tokens": tokens, "ner_tags": ner_tags})
        
    return all_output_data



# Example usage for the first entry
input_data_1 = json.load(open("/Users/davidebuoso/Desktop/dev/dnlp/L-NER/legal_ner/data/NER_TRAIN/NER_TRAIN_ALL.json"))

output_data_1 = convert_format(input_data_1)
with open("legal_ner/data/NER_TRAIN/NER_TRAIN_ALL_OT.json", "w") as f:
    for line in output_data_1:
        f.write(str(line)+"\n")
