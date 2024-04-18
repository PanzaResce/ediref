import torch
from torch import nn
from transformers import LlamaTokenizer, LlamaForSequenceClassification, LlamaModel, BertPreTrainedModel, BertConfig, BertModel

class LLAMA_EFR(nn.Module):
    def __init__(self, model_path, hid_dim=3200):
        self.core = LlamaModel.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map='auto', offload_folder="offload_base", offload_state_dict=True)
        
        self.trig_score1 = nn.Linear(hid_dim, hid_dim, bias=True, dtype=torch.float16).to("cuda")
        self.trig_score2 = nn.Linear(hid_dim, 1, bias=False, dtype=torch.float16).to("cuda")
    
    def forward(self, d_ids):
        pass


class BERT_Model_Phrase_Concatenation(BertPreTrainedModel):
    def __init__(self, df_manager, load=None, pos_weight=None, freeze=False, model_card='bert-base-uncased'):
        self.config = BertConfig.from_pretrained(model_card, output_attentions=True, output_hidden_states=True)
        self.freeze = freeze

        super().__init__(self.config)
        self.core = self.initialize_model(load)
        # Freeze BERT embedding layer parameters
        if freeze:
            for param in self.core.embeddings.parameters():
                param.requires_grad = False

        if pos_weight == None:
            self.pos_weight = torch.ones([self.config.num_labels]).to("cuda")
        else:
            self.pos_weight = pos_weight
        self.emotion_head = nn.Linear(self.config.hidden_size, len(df_manager.unique_emotions))
        self.trigger_head = nn.Linear(self.config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.post_init()

    def initialize_model(self, load):
        if load == None:
            return BertModel(self.config)
        else:
            print("load = ", load)
            return BertModel.from_pretrained(load, config=load+'/config.json', local_files_only=True)
        
    def forward(
        self,
        utterance_ids=None,
        utterance_mask=None,
        dialogue_ids=None,
        dialogue_mask=None,
        token_type_ids=None,
        emotions_id_one_hot_encoding=None, 
        emotions_id=None,
        triggers=None,
        dialogue_index=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.core(
            input_ids=dialogue_ids,
            attention_mask=dialogue_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        sequence_output = outputs.pooler_output.unsqueeze(1)
        sentence_output = self.core(
            input_ids=utterance_ids,
            attention_mask=utterance_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        model_output = torch.cat((sequence_output, sentence_output.last_hidden_state), dim=1)
        emotion_logits = torch.mean(self.emotion_head(model_output), dim=(1))
        trigger_logits = torch.mean(self.trigger_head(model_output), dim=(1))
        emotion_logits = self.softmax(emotion_logits)
        trigger_logits = torch.sigmoid(trigger_logits)
        return {"emotion_logits": emotion_logits,
                "trigger_logits": trigger_logits}
    
    