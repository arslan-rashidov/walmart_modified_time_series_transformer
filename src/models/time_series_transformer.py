import torch
from torch import nn, Tensor, IntTensor


class FeedForwardLayer(nn.Module):
    def __init__(
            self,
            input_size=1,
            hidden_size=32,
            output_size=32
    ):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            tabular_cat_features_size: int,  # 2
            tabular_cat_features_possible_nums: IntTensor,  # [45, 81]

            tabular_num_features_size: int,  # 1
            tabular_num_features_ffn_hidden_size: int,  # 32

            time_series_cat_features_size: int,  # 1
            time_series_cat_features_possible_nums: IntTensor,  # [2]

            time_series_size: int,  # 10
            time_series_numerical_features_size: int,  # 5

            d_model: int = 1024,

    ):
        super().__init__()

        self.tabular_cat_features_embeddings_table = torch.nn.ModuleList(
            [torch.nn.Embedding(num_embeddings=tabular_cat_features_possible_nums[cat_feature_i],
                                embedding_dim=d_model)
             for cat_feature_i in range(tabular_cat_features_size)]
        )

        self.tabular_cat_features_pos_embeddings_table = nn.Embedding(num_embeddings=tabular_cat_features_size + 1,
                                                                      embedding_dim=d_model)

        self.tabular_num_features_ffn = FeedForwardLayer(input_size=tabular_num_features_size,
                                                         hidden_size=tabular_num_features_ffn_hidden_size,
                                                         output_size=d_model)

        self.time_series_cat_features_embeddings_table = nn.ModuleList(
            [nn.Embedding(num_embeddings=time_series_cat_features_possible_nums[cat_feature_i],
                          embedding_dim=d_model)
             for cat_feature_i in range(time_series_cat_features_size)]
        )

        self.time_series_ffn = FeedForwardLayer(input_size=1, output_size=d_model)
        self.time_series_numerical_features = FeedForwardLayer(input_size=time_series_numerical_features_size,
                                                               output_size=d_model)

        self.tabular_cat_features_size = tabular_cat_features_size
        self.tabular_num_features_size = tabular_num_features_size
        self.tabular_features_size = tabular_cat_features_size + tabular_num_features_size

        self.time_series_cat_features_size = time_series_cat_features_size
        self.time_series_size = time_series_size

        self.positional_embedding_table = torch.nn.Embedding(num_embeddings=self.time_series_size,
                                                             embedding_dim=d_model)

        self.d_model = d_model

    def forward(
            self,
            time_series_num_features: Tensor,
            tabular_cat_features: Tensor = None,
            tabular_num_features: Tensor = None,
            time_series_cat_features: Tensor = None
    ):
        batch_size = time_series_num_features.shape[0]

        if tabular_cat_features is not None:
            tabular_cat_features_embeddings = torch.sum(torch.stack([self.tabular_cat_features_embeddings_table[
                                                                         cat_feature_i](
                tabular_cat_features[:, cat_feature_i].long())
                                                                     for cat_feature_i in
                                                                     range(tabular_cat_features.shape[1])], dim=1),
                                                        dim=1)

        if tabular_num_features is not None:
            tabular_num_features_embeddings = self.tabular_num_features_ffn(tabular_num_features)

        if time_series_cat_features is not None:
            time_series_cat_features = time_series_cat_features.reshape(batch_size, self.time_series_size, -1)
            time_series_cat_features_embeddings = torch.cat(
                [self.time_series_cat_features_embeddings_table[cat_feature_i](
                    time_series_cat_features[:, :, cat_feature_i].long())
                    for cat_feature_i in
                    range(time_series_cat_features.shape[-1])], dim=-1)

        time_series = time_series_num_features[:, :, 0].reshape(batch_size, self.time_series_size, -1)
        time_series_num_features = time_series_num_features[:, :, 1:]

        time_series_embeddings = self.time_series_ffn(time_series)
        time_series_num_features_embeddings = self.time_series_numerical_features(time_series_num_features)

        tabular_embeddings = torch.sum(
            torch.stack([tabular_cat_features_embeddings, tabular_num_features_embeddings], dim=0), dim=0)
        time_series_embeddings = torch.sum(torch.stack(
            [time_series_cat_features_embeddings, time_series_num_features_embeddings, time_series_embeddings]), dim=0)

        for step_i in range(len(time_series_embeddings[1])):
            time_series_embeddings[:, step_i] += self.positional_embedding_table(
                torch.tensor([step_i]))

            #time_series_embeddings[:, step_i] += self.positional_embedding_table(
            #    torch.tensor([step_i], device='cuda:0'))

        tabular_embeddings = tabular_embeddings.reshape(batch_size, -1, self.d_model).repeat(1, self.time_series_size,
                                                                                             1)

        time_series_embeddings = time_series_embeddings.reshape(batch_size, self.time_series_size, -1)

        return torch.sum(torch.stack([tabular_embeddings, time_series_embeddings]), dim=0)


class TimeSeriesTransformer(nn.Module):
    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".

    A detailed description of the code can be found in my article here:

    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.

    Unlike the paper, this class assumes that input layers, positional encoding
    layers and linear mapping layers are separate from the encoder and decoder,
    i.e. the encoder and decoder only do what is depicted as their sub-layers
    in the paper. For practical purposes, this assumption does not make a
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the
    Encoder() and Decoder() classes.

    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020).
    'Deep Transformer Models for Time Series Forecasting:
    The Influenza Prevalence Case'.
    arXiv:2001.08317 [cs, stat] [Preprint].
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).

    [2] Vaswani, A. et al. (2017)
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint].
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).

    """

    def __init__(self,
                 tabular_cat_features_size: int,  # 2
                 tabular_cat_features_possible_nums: IntTensor,  # [45, 81]

                 tabular_num_features_size: int,  # 1
                 tabular_num_features_ffn_hidden_size: int,  # 32

                 time_series_cat_features_size: int,  # 1
                 time_series_cat_features_possible_nums: IntTensor,  # [2]

                 time_series_numerical_features_size: int,  # 5

                 enc_seq_len: int,  # 10

                 dec_seq_len: int = 4,
                 batch_first: bool = True,
                 out_seq_len: int = 3,
                 dim_val: int = 512,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_decoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 1
                 ):
        """
        Args:

            input_size: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder

            dim_val: int, aka d_model. All sub-layers in the model produce
                     outputs of dimension dim_val

            n_encoder_layers: int, number of stacked encoder layers in the encoder

            n_decoder_layers: int, number of stacked encoder layers in the decoder

            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder

            dropout_decoder: float, the dropout rate of the decoder

            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer
                                     of the encoder

            dim_feedforward_decoder: int, number of neurons in the linear layer
                                     of the decoder

            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__()

        self.dec_seq_len = dec_seq_len

        # print("input_size is: {}".format(input_size))
        # print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_embedding_layer = EmbeddingLayer(
            tabular_cat_features_size=tabular_cat_features_size,
            tabular_cat_features_possible_nums=tabular_cat_features_possible_nums,
            tabular_num_features_size=tabular_num_features_size,
            tabular_num_features_ffn_hidden_size=tabular_num_features_ffn_hidden_size,
            time_series_cat_features_size=time_series_cat_features_size,
            time_series_cat_features_possible_nums=time_series_cat_features_possible_nums,
            time_series_size=enc_seq_len,
            time_series_numerical_features_size=time_series_numerical_features_size,
            d_model=dim_val
        )

        self.decoder_embedding_layer = EmbeddingLayer(
            tabular_cat_features_size=tabular_cat_features_size,
            tabular_cat_features_possible_nums=tabular_cat_features_possible_nums,
            tabular_num_features_size=tabular_num_features_size,
            tabular_num_features_ffn_hidden_size=tabular_num_features_ffn_hidden_size,
            time_series_cat_features_size=time_series_cat_features_size,
            time_series_cat_features_possible_nums=time_series_cat_features_possible_nums,
            time_series_size=dec_seq_len,
            time_series_numerical_features_size=time_series_numerical_features_size,
            d_model=dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
        )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

    def forward(self,
                tabular_categorical_features,
                tabular_numerical_features,
                src_time_series_categorical_features,
                src_time_series_numerical_features,
                trg_time_series_categorical_features,
                trg_time_series_numerical_features,
                src_mask: Tensor = None,
                tgt_mask: Tensor = None) -> Tensor:
        """
        Returns a tensor of shape:

        [target_sequence_length, batch_size, num_predicted_features]

        Args:

            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)

            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)

            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence

            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence


        """

        # print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        # print("From model.forward(): tgt size = {}".format(tgt.size()))

        # Pass throguh the input layer right before the encoder
        src = self.encoder_embedding_layer(
            tabular_cat_features=tabular_categorical_features,
            tabular_num_features=tabular_numerical_features,
            time_series_cat_features=src_time_series_categorical_features,
            time_series_num_features=src_time_series_numerical_features
        )  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder(  # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
        )
        # print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_embedding_layer(
            tabular_cat_features=tabular_categorical_features,
            tabular_num_features=tabular_numerical_features,
            time_series_cat_features=trg_time_series_categorical_features,
            time_series_num_features=trg_time_series_numerical_features
        )  # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        # print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

        # if src_mask is not None:
        # print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        # if tgt_mask is not None:
        # print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output)  # shape [batch_size, target seq len]
        # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))

        return decoder_output


