import json
def filter_entries_by_labels(input_json, output_json, target_labels):
    with open(input_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    filtered_data = []

    for entry in data:
        annotations = entry.get('annotations', [])
        labels = set()
        for annotation in annotations:
            for result in annotation.get('result', []):
                labels.update(result.get('value', {}).get('labels', []))

        if labels.intersection(target_labels):
            filtered_data.append(entry)

    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_json_path = "legal_ner/data/NER_TRAIN/NER_TRAIN_ALL.json"
    output_json_path = "legal_ner/data/NER_TRAIN/NER_TRAIN_SMALL.json"

    
    target_labels = {"ORG", "GPE", "PRECEDENT"}  # Replace with your target labels

    filter_entries_by_labels(input_json_path, output_json_path, target_labels)

    input_json_path = "legal_ner/data/NER_DEV/NER_DEV_ALL.json"
    output_json_path = "legal_ner/data/NER_DEV/NER_DEV_SMALL.json"

    filter_entries_by_labels(input_json_path, output_json_path, target_labels)
