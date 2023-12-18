from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import transformers
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b-slimpj")
print(model)