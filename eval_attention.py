from torch.autograd import Variable
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn



class SCAttention(nn.Module) :

    def __init__(self, input_size, hidden_size) :
        super(SCAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.map_linear = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self) :
        nn.init.xavier_uniform_(self.W.weight.data)
        self.W.bias.data.fill_(0.1)

    def forward(self, passage, question, q_mask):
        # Wp = F.relu(self.W(passage))
        # Wq = F.relu(self.W(question))
        # no relu
        Wp = self.W(passage)    # [bsz,n,emd]
        Wq = self.W(question)   # [bsz,m,emd]
        scores = torch.bmm(Wp, Wq.transpose(2, 1))     #[bsz,n,m]
        mask = q_mask.unsqueeze(1).repeat(1, passage.size(1), 1)   #[bsz,m] --> [bsz,n,m]
        # scores.data.masked_fill_(mask.data, -float('inf'))
        alpha = masked_softmax(scores, mask)  # [bsz,n,m]
        output = torch.bmm(alpha, Wq)  # [bsz,m,emd]
        output = nn.ReLU()(self.map_linear(output))
        return output    # 实质是A关于B计算attention后的增量

def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
    return result

# questions = np.random.randn(32, 384, 1024) # batch_size （question_num) , len of question, dim
# facts = np.random.randn(32, 15, 1024)  # batch_size （fac_num) , len of fact, dim
# questions = torch.Tensor(questions)
# facts = torch.Tensor(facts)
# fact_mask = torch.Tensor(32, 15).fill_(1)  # global attention
#
# sc_attention = SCAttention(1024, 1024)  # input size --> hidden size
# output = sc_attention(questions, facts, fact_mask)
# questions = questions + output  # batch_size （question_num) , dim
# print(questions.size())