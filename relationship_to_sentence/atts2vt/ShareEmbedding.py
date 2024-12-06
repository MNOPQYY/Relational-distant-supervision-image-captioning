import torch
import torch.nn as nn

class Share_Embedding(nn.Module):
    def __init__(self, opt, vocab_size):
        super(Share_Embedding, self).__init__()
        
        # self.embed_dim = opt.embed_dim
        self.embed_dim = 512
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    def forward(self, input_sequence, sinlge_word=True, rel_word_num=None):
        result = self.embedding(input_sequence)
        if not sinlge_word:
            result_rel = torch.FloatTensor(input_sequence.size(0),input_sequence.size(1),self.embed_dim)
            for bs_id, rel_bs in enumerate(result):
                for rel_id, rel in enumerate(rel_bs):
                    
                    result_rel[bs_id,rel_id] = torch.mean(rel[:rel_word_num[bs_id,rel_id]])
            result = result_rel.cuda()
            # result = torch.mean(result,dim=2)
        return result