# neural.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Main MLP
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, first_hidden_size, other_hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.first_hidden_size = first_hidden_size
        self.other_hidden_size = other_hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, first_hidden_size)
        self.fc2 = nn.Linear(first_hidden_size, other_hidden_size)
        self.fc3 = nn.Linear(other_hidden_size, other_hidden_size)
        self.fc4 = nn.Linear(other_hidden_size, other_hidden_size)
        self.fc5 = nn.Linear(other_hidden_size, output_size)

        self.fc_gate = nn.Linear(other_hidden_size, other_hidden_size)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    # r-skip + r-gate hybrid forward
    def forward(self, x):
        # initial projection
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))

        # R-gate + R-skip block
        for _ in range(3):
            # gate ∈ (0,1)
            g = torch.sigmoid(self.fc_gate(h))
            # recursive update
            u = F.silu(self.fc3(h))
            # gated residual: h ← h + g·(u−h)
            h = h + g * (u - h)

        # extra residual refinement
        h4 = F.silu(self.fc4(h))
        h = h + h4

        return self.fc5(h)

    # ------------------------------------------------------------
    # Network Scaling
    # ------------------------------------------------------------

    def upscale(self, new_hidden_size):
        """
        Return a new NeuralNetwork with larger hidden size,
        embedding old weights conservatively so training can continue
        without resetting the cognitive manifold.
        """

        old_first = self.first_hidden_size
        old_other = self.other_hidden_size

        # nothing to do if size is same or smaller
        if new_hidden_size <= old_other:
            raise ValueError("new_hidden_size must be larger than current hidden size")

        # create new model
        new_model = NeuralNetwork(
            input_size=self.input_size,
            first_hidden_size=new_hidden_size,
            other_hidden_size=new_hidden_size,
            output_size=self.output_size
        )

        # helper: embed old matrix into new larger matrix
        def widen_linear(old_layer, new_layer, old_in, old_out, new_in, new_out):
            """
            Embed old weights into top-left corner.
            New weights initialized extremely small (epsilon).
            """
            eps = 1e-5

            # new weight matrix
            W = new_layer.weight.data
            W.zero_()

            # copy old block
            W[:old_out, :old_in] = old_layer.weight.data

            # tiny noise for new parameters
            W[old_out:, :] = eps * torch.randn(new_out - old_out, new_in)
            W[:, old_in:] = eps * torch.randn(new_out, new_in - old_in)

            # new bias
            b = new_layer.bias.data
            b.zero_()
            b[:old_out] = old_layer.bias.data
            b[old_out:] = eps * torch.randn(new_out - old_out)

        # widen fc1: input_size → first_hidden_size
        widen_linear(
            self.fc1, new_model.fc1,
            old_in=self.input_size,
            old_out=old_first,
            new_in=self.input_size,
            new_out=new_hidden_size
        )

        # widen fc2: first_hidden_size → other_hidden_size
        widen_linear(
            self.fc2, new_model.fc2,
            old_in=old_first,
            old_out=old_other,
            new_in=new_hidden_size,
            new_out=new_hidden_size
        )

        # widen fc3, fc4, fc_gate: other_hidden_size → other_hidden_size
        for old_layer, new_layer in [
            (self.fc3, new_model.fc3),
            (self.fc4, new_model.fc4),
            (self.fc_gate, new_model.fc_gate)
        ]:
            widen_linear(
                old_layer, new_layer,
                old_in=old_other,
                old_out=old_other,
                new_in=new_hidden_size,
                new_out=new_hidden_size
            )

        # widen fc5: other_hidden_size → output_size
        eps = 1e-5
        new_model.fc5.weight.data.zero_()
        new_model.fc5.weight.data[:, :old_other] = self.fc5.weight.data
        new_model.fc5.weight.data[:, old_other:] = eps * torch.randn(self.output_size, new_hidden_size - old_other)

        new_model.fc5.bias.data = self.fc5.bias.data.clone()

        return new_model

    def downscale(self, new_hidden_size):
        """
        Return a new NeuralNetwork with smaller hidden size,
        projecting the large model onto a smaller core manifold.
        """

        old_first = self.first_hidden_size
        old_other = self.other_hidden_size

        if new_hidden_size >= old_other:
            raise ValueError("new_hidden_size must be smaller than current hidden size")

        # new model with smaller hidden size
        new_model = NeuralNetwork(
            input_size=self.input_size,
            first_hidden_size=new_hidden_size,
            other_hidden_size=new_hidden_size,
            output_size=self.output_size
        )

        # helper: project big layer to smaller layer
        def project_linear(old_layer, new_layer, old_in, old_out, new_in, new_out):
            """
            Neem de 'core' S-dimensies uit de grote matrix.
            We gebruiken gewoon de eerste new_out x new_in blok.
            """
            W_old = old_layer.weight.data
            b_old = old_layer.bias.data

            W_new = new_layer.weight.data
            b_new = new_layer.bias.data

            # weight: take the first new_out x new_in
            W_new.copy_(W_old[:new_out, :new_in])
            # bias: first new_out
            b_new.copy_(b_old[:new_out])

        # fc1: input_size → first_hidden_size
        project_linear(
            self.fc1, new_model.fc1,
            old_in=self.input_size,
            old_out=old_first,
            new_in=self.input_size,
            new_out=new_hidden_size
        )

        # fc2: first_hidden_size → other_hidden_size
        project_linear(
            self.fc2, new_model.fc2,
            old_in=old_first,
            old_out=old_other,
            new_in=new_hidden_size,
            new_out=new_hidden_size
        )

        # fc3, fc4, fc_gate: other_hidden_size → other_hidden_size
        for old_layer, new_layer in [
            (self.fc3, new_model.fc3),
            (self.fc4, new_model.fc4),
            (self.fc_gate, new_model.fc_gate)
        ]:
            project_linear(
                old_layer, new_layer,
                old_in=old_other,
                old_out=old_other,
                new_in=new_hidden_size,
                new_out=new_hidden_size
            )

        # fc5: other_hidden_size → output_size
        W5_old = self.fc5.weight.data
        b5_old = self.fc5.bias.data

        W5_new = new_model.fc5.weight.data
        b5_new = new_model.fc5.bias.data

        # take only the first new_hidden_size columns
        W5_new.copy_(W5_old[:, :new_hidden_size])
        b5_new.copy_(b5_old)

        return new_model
