import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import LukeModel
from transformers.models.luke.configuration_luke import LukeConfig

# Define an Expert Layer
class NERLukeExpert(nn.Module):
    def __init__(self, config):
        super(NERLukeExpert, self).__init__()
        self.luke = LukeModel(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.luke(input_ids=input_ids, attention_mask=attention_mask)
        # Assuming you want to use the last layer's embeddings
        embeddings = outputs.last_hidden_state
        return embeddings

# Define the Dynamic Mixture of Experts Model
class DynamicMoENerLukeModel(nn.Module):
    def __init__(self, config, num_experts, num_labels):
        super(DynamicMoENerLukeModel, self).__init__()
        self.experts = nn.ModuleList([NERLukeExpert(config) for _ in range(num_experts)])
        self.gating_network = nn.Linear(config.hidden_size, num_experts)
        self.final_classifier = nn.Linear(config.hidden_size * num_experts, num_labels)

    def forward(self, input_ids, attention_mask):
        expert_outputs = [expert(input_ids, attention_mask) for expert in self.experts]
        expert_embeddings = torch.cat(expert_outputs, dim=2)

        gating_scores = softmax(self.gating_network(expert_embeddings), dim=-1).unsqueeze(-1)

        # Weighted sum of expert outputs based on gating scores
        weighted_expert_outputs = expert_embeddings * gating_scores
        final_output = torch.sum(weighted_expert_outputs, dim=2)

        # Apply a classifier on the final weighted embeddings
        logits = self.final_classifier(final_output)

        return logits, gating_scores

# Example usage
config = LukeConfig.from_pretrained("studio-ousia/luke-base")
num_experts = 3   # You can adjust the number of experts
original_label_list = [
        "COURT",
        "PETITIONER",
        "RESPONDENT",
        "JUDGE",
        "DATE",
        "ORG",
        "GPE",
        "STATUTE",
        "PROVISION",
        "PRECEDENT",
        "CASE_NUMBER",
        "WITNESS",
        "OTHER_PERSON",
        "LAWYER"
    ]
labels_list = ["B-" + l for l in original_label_list]
labels_list += ["I-" + l for l in original_label_list]
num_labels = len(labels_list) + 1  # Assuming labels_list is defined in your code

dynamic_moe_model = DynamicMoENerLukeModel(config, num_experts, num_labels)

# Assuming you have input_ids and attention_mask as input tensors
input_ids_1 = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])  # Replace with your actual values
attention_mask_1 = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])  # Replace with your actual values

# Sample input_ids and attention_mask for instance 2
input_ids_2 = torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])  # Replace with your actual values
attention_mask_2 = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])  # Replace with your actual values

# Example usage of the model
logits_1, gating_scores_1 = dynamic_moe_model(input_ids_1, attention_mask_1)
logits_2, gating_scores_2 = dynamic_moe_model(input_ids_2, attention_mask_2)

# Printing the results for instance 1
print("Logits for Instance 1:")
print(logits_1)
print("\nGating Scores for Instance 1:")
print(gating_scores_1)
