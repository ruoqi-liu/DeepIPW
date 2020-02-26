import torch
import numpy as np

class LSTMModel(torch.nn.Module):
    def __init__(self, diag_vocab_size, med_vocab_size, diag_embedding_size,
                 med_embedding_size, diag_hidden_size, med_hidden_size,hidden_size,
                 end_index, pad_index, bidirectional=True):
        super().__init__()
        self.pad_index = pad_index
        self.end_index = end_index
        self.diag_embedding = torch.nn.Linear(diag_vocab_size, diag_embedding_size, bias=False)
        self.med_embedding = torch.nn.Linear(med_vocab_size, med_embedding_size, bias=False)

        self.diag_encoder = torch.nn.LSTM(diag_embedding_size, diag_hidden_size, batch_first=True,
                                          bidirectional=bidirectional)

        self.med_encoder = torch.nn.LSTM(med_embedding_size, med_hidden_size, batch_first=True,
                                          bidirectional=bidirectional)

        if bidirectional:
            diag_hidden_size = diag_hidden_size * 2
            med_hidden_size=med_hidden_size*2

        self.attention_diag_encoder=torch.nn.Sequential(
            torch.nn.Linear(diag_hidden_size, 1),
            torch.nn.Tanh(),
        )
        self.attention_med_encoder = torch.nn.Sequential(
            torch.nn.Linear(med_hidden_size, 1),
            torch.nn.Tanh(),
        )

        # IPW
        self.lstm2hidden_ipw= torch.nn.Sequential(
            torch.nn.Linear(med_hidden_size+diag_hidden_size+2, hidden_size),
            torch.nn.ReLU(),
        )
        self.hidden2out_ipw = torch.nn.Linear(hidden_size, 1, bias=False)

        # Outcome
        # self.lstm2hidden_outcome = torch.nn.Sequential(
        #     torch.nn.Linear(med_hidden_size + diag_hidden_size + 3, hidden_size),
        #     torch.nn.ReLU(),
        # )
        # self.hidden2out_outcome = torch.nn.Linear(hidden_size, 1, bias=False)

    def softmax_masked(self,inputs, mask, dim=1, epsilon=0.0000001):
        inputs_exp=torch.exp(inputs)
        inputs_exp=inputs_exp * mask.float()
        inputs_exp_sum=inputs_exp.sum(dim=dim)
        inputs_attention=inputs_exp / (inputs_exp_sum.unsqueeze(dim)+epsilon)

        return inputs_attention

    def diag_encode(self, inputs):
        inputs_mask=(inputs.sum(dim=-1)!=0).long()
        inputs_emb = self.diag_embedding(inputs.float())
        outputs, (h, c) = self.diag_encoder(inputs_emb)

        att_enc=self.attention_diag_encoder(outputs).squeeze(-1)
        att_normalized=self.softmax_masked(att_enc,inputs_mask)
        hidden=torch.sum(outputs*att_normalized.unsqueeze(-1),dim=1)
        original = torch.sum(inputs.float()*att_normalized.unsqueeze(-1),dim=1)

        return hidden, original

    def med_encode(self, inputs):
        inputs_mask=(inputs.sum(dim=-1)!=0).long()
        inputs_emb = self.med_embedding(inputs.float())
        outputs, (h, c) = self.med_encoder(inputs_emb)

        att_enc=self.attention_med_encoder(outputs).squeeze(-1)
        att_normalized=self.softmax_masked(att_enc,inputs_mask)
        hidden=torch.sum(outputs*att_normalized.unsqueeze(-1),dim=1)
        original = torch.sum(inputs.float() * att_normalized.unsqueeze(-1), dim=1)

        return hidden, original

    def forward(self, confounder):
        diag_inputs, med_inputs, sexes, ages=confounder
        diag_hidden, diag_original = self.diag_encode(diag_inputs)
        med_hidden, med_original = self.med_encode(med_inputs)
        original = torch.cat((diag_original,med_original, sexes.float().view(sexes.size(0),1), ages.float().view(ages.size(0),1)),dim=1)

        # IPW
        hidden=torch.cat((diag_hidden,med_hidden, sexes.float().view(sexes.size(0),1), ages.float().view(ages.size(0),1)),dim=1)
        hidden=self.lstm2hidden_ipw(hidden)
        outputs_logits_ipw = self.hidden2out_ipw(hidden)


        return outputs_logits_ipw.view(outputs_logits_ipw.size(0)), original