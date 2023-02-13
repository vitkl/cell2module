from typing import Optional

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer.autoguide.utils import deep_getattr, deep_setattr
from pyro.nn import PyroModule
from scvi import REGISTRY_KEYS
from scvi.nn import one_hot


class MultimodalNMFPyroModel(PyroModule):
    r"""
    Cell2module model

    Cell2module model treats RNA count data :math:`D` as Poisson distributed,
    given transcription rate :math:`\\mu_{c,g}` and a range of variables accounting for technical effects:

    .. math::
        D_{c,g} \\sim \\mathtt{NB}(\\mu=\\mu_{c,g}, \alpha_{a,g})
    .. math::
        \\mu_{c,g} = ((\\sum_f w_{c,f} g_{f,g}) + s_{e,g}) * y_c * y_{t,g}

    .. math::
        D_{c,r} \\sim \\mathtt{Binomial}(n=1, p=p_{c,r})
    .. math::
        p_{c,r} = (\\sum_f w_{c,f}, g_{f,r}) * y^{chr}_c * y_{t,r}


    Here, :math:`\\mu_{c,g}` denotes expected RNA count :math:`g` in each cell :math:`c`;
    :math:`\alpha_{a,g}` denotes per gene :math:`g` stochatic/unexplained overdispersion for each covariate :math:`a`;
    :math:`w_{c,f}` denotes cell loadings of each factor :math:`f` for each cell :math:`c`;
    :math:`g_{f,g}` denotes gene loadings of each factor :math:`f` for each gene :math:`g`;
    :math:`s_{e,g}` denotes additive background for each gene :math:`g` and for each experiment :math:`e`,
        to account for contaminating RNA;
    :math:`y_c` denotes normalisation for each cell :math:`c` with a prior mean for each experiment :math:`e`, to account for RNA detection sensitivity, sequencing depth;
    :math:`y_{t,g}` denotes per gene :math:`g` detection efficiency normalisation for each technology :math:`t`;
    :math:`p_{c,r}` denotes accessibility probability of each genome region :math:`r` in each
      cell :math:`c`;
    :math:`g_{f,r}` denotes loadings of each factor :math:`f` for each genome region :math:`r`;
    :math:`y^{chr}_c` denotes normalisation for each cell :math:`c` with a prior mean for each experiment :math:`e`,
        to account for chromatin assay efficiency, sequencing depth;
    :math:`y_{t,r}` denotes per genome region :math:`r` detection efficiency normalisation for each technology :math:`t`;
    """

    def __init__(
        self,
        n_obs,
        n_vars,
        n_factors,
        n_batch,
        n_extra_categoricals,
        n_var_categoricals,
        gene_bool: np.array,
        n_labels: int = 0,
        factor_number_prior={"factors_per_cell": 10.0, "mean_var_ratio": 1.0},
        factor_prior={
            "rate": 1.0,
            "alpha": 1.0,
            "modules_per_gene": 10.0,
            "modules_per_region": 10.0,
        },
        stochastic_v_ag_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_alpha_hyp_prior={"alpha": 9.0, "beta": 3.0},
        gene_add_mean_hyp_prior={
            "alpha": 1.0,
            "beta": 100.0,
        },
        detection_hyp_prior={"alpha": 200.0, "mean_alpha": 1.0, "mean_beta": 1.0},
        gene_tech_prior={"mean": 1.0, "alpha": 1000.0},
        region_add_prior={"alpha": 1.0, "beta": 100.0},
        binomial_n: float = 1,
        fixed_vals: Optional[dict] = None,
        init_vals: Optional[dict] = None,
        init_alpha=10.0,
        rna_model: bool = True,
        chromatin_model: bool = False,
        use_binary_chromatin: bool = False,
        use_orthogonality_constraint: bool = False,
        use_amortised_cell_loadings_as_prior: bool = False,
        use_non_linear_decoder: bool = False,
        n_layers=2,
        n_hidden=256,
        decoder_bias=True,
        dropout_rate=0.1,
    ):
        """
        Create a Cell2module model.

        Parameters
        ----------
        n_obs
        n_vars
        n_factors
        n_batch
        gene_bool
        stochastic_v_ag_hyp_prior
        gene_add_alpha_hyp_prior
        gene_add_mean_hyp_prior
        detection_hyp_prior
        gene_tech_prior
        """

        ############# Initialise parameters ################
        super().__init__()

        self.rna_model = rna_model
        self.chromatin_model = chromatin_model

        self.weights = PyroModule()

        self.n_obs = n_obs
        self.n_vars = n_vars
        self.n_factors = n_factors
        self.n_batch = n_batch
        self.n_extra_categoricals = n_extra_categoricals
        self.n_var_categoricals = n_var_categoricals
        self.n_labels = n_labels

        self.use_binary_chromatin = use_binary_chromatin
        self.use_orthogonality_constraint = use_orthogonality_constraint
        self.use_amortised_cell_loadings_as_prior = use_amortised_cell_loadings_as_prior
        self.use_non_linear_decoder = use_non_linear_decoder
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.decoder_bias = decoder_bias
        self.dropout_rate = dropout_rate

        self.gene_bool = gene_bool.astype(int).flatten()
        self.gene_ind = np.where(gene_bool)[0]
        self.n_genes = len(self.gene_ind)
        self.register_buffer("gene_ind_tt", torch.tensor(self.gene_ind))

        self.region_ind = np.where(np.logical_not(gene_bool))[0]
        self.n_regions = self.n_vars - self.n_genes
        self.register_buffer("region_ind_tt", torch.tensor(self.region_ind))

        # RNA model priors
        self.stochastic_v_ag_hyp_prior = stochastic_v_ag_hyp_prior
        self.gene_add_alpha_hyp_prior = gene_add_alpha_hyp_prior
        self.gene_add_mean_hyp_prior = gene_add_mean_hyp_prior
        self.gene_tech_prior = gene_tech_prior

        # Chromatin model priors
        self.region_add_prior = region_add_prior
        self.binomial_n = binomial_n

        # Shared priors
        self.detection_hyp_prior = detection_hyp_prior

        self.factor_number_prior = factor_number_prior
        self.factor_prior = factor_prior

        # Fixed values (gene loadings or cell loadings)
        if (fixed_vals is not None) & (type(fixed_vals) is dict):
            self.np_fixed_vals = fixed_vals
            for k in fixed_vals.keys():
                self.register_buffer(f"fixed_val_{k}", torch.tensor(fixed_vals[k]))
            if "n_factors" in fixed_vals.keys():
                self.n_factors = fixed_vals["n_factors"]
            if "g_fg" in fixed_vals.keys():
                self.n_genes = fixed_vals["g_fg"].shape[1]

        # Initial values
        if (init_vals is not None) & (type(init_vals) is dict):
            self.np_init_vals = init_vals
            for k in init_vals.keys():
                self.register_buffer(f"init_val_{k}", torch.tensor(init_vals[k]))
            self.init_alpha = init_alpha
            self.register_buffer("init_alpha_tt", torch.tensor(self.init_alpha))

        # Shared priors
        self.register_buffer(
            "detection_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_alpha",
            torch.tensor(self.detection_hyp_prior["mean_alpha"]),
        )
        self.register_buffer(
            "detection_mean_hyp_prior_beta",
            torch.tensor(self.detection_hyp_prior["mean_beta"]),
        )

        # per cell module activity priors
        self.register_buffer(
            "factors_per_cell_alpha",
            torch.tensor(self.factor_number_prior["factors_per_cell"])
            * torch.tensor(self.factor_number_prior["mean_var_ratio"]),
        )
        self.register_buffer(
            "factors_per_cell_beta",
            torch.tensor(self.factor_number_prior["mean_var_ratio"]),
        )

        # per gene rate priors
        self.register_buffer(
            "factor_prior_alpha",
            torch.tensor(self.factor_prior["alpha"]),
        )
        self.register_buffer(
            "factor_prior_beta",
            torch.tensor(self.factor_prior["alpha"] / self.factor_prior["rate"]),
        )

        # RNA model priors
        self.register_buffer(
            "modules_per_gene",
            torch.tensor(self.factor_prior["modules_per_gene"]),
        )

        self.register_buffer(
            "gene_tech_prior_alpha",
            torch.tensor(self.gene_tech_prior["alpha"]),
        )
        self.register_buffer(
            "gene_tech_prior_beta",
            torch.tensor(self.gene_tech_prior["alpha"] / self.gene_tech_prior["mean"]),
        )

        self.register_buffer(
            "stochastic_v_ag_hyp_prior_alpha",
            torch.tensor(self.stochastic_v_ag_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "stochastic_v_ag_hyp_prior_beta",
            torch.tensor(self.stochastic_v_ag_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_alpha",
            torch.tensor(self.gene_add_alpha_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_alpha_hyp_prior_beta",
            torch.tensor(self.gene_add_alpha_hyp_prior["beta"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_alpha",
            torch.tensor(self.gene_add_mean_hyp_prior["alpha"]),
        )
        self.register_buffer(
            "gene_add_mean_hyp_prior_beta",
            torch.tensor(self.gene_add_mean_hyp_prior["beta"]),
        )

        # Chromatin model priors
        self.register_buffer(
            "modules_per_region",
            torch.tensor(self.factor_prior["modules_per_region"]),
        )

        self.register_buffer(
            "region_add_prior_alpha",
            torch.tensor(self.region_add_prior["alpha"]),
        )
        self.register_buffer(
            "region_add_prior_beta",
            torch.tensor(self.region_add_prior["beta"]),
        )

        self.register_buffer("binomial_n_tt", torch.tensor(binomial_n))

        self.register_buffer("ones", torch.ones((1, 1)))
        self.register_buffer("zeros", torch.zeros((1, 1)))
        self.register_buffer("ten", torch.tensor(10.0))
        self.register_buffer("n_factors_torch", torch.tensor(float(self.n_factors)))
        self.register_buffer("ones_1_n_factors", torch.ones((1, self.n_factors)))
        self.register_buffer("eps", torch.tensor(1e-8))

    ############# Define the model ################
    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x_data = tensor_dict[REGISTRY_KEYS.X_KEY]
        ind_x = tensor_dict["ind_x"].long().squeeze()
        batch_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        if REGISTRY_KEYS.LABELS_KEY in tensor_dict.keys():
            label_index = tensor_dict[REGISTRY_KEYS.LABELS_KEY]
        else:
            label_index = tensor_dict[REGISTRY_KEYS.BATCH_KEY]
        rna_index = tensor_dict["rna_index"].bool()
        chr_index = tensor_dict["chr_index"].bool()
        extra_categoricals = tensor_dict[REGISTRY_KEYS.CAT_COVS_KEY]
        var_categoricals = tensor_dict["var_categoricals"].long()
        return (
            x_data,
            ind_x,
            batch_index,
            label_index,
            rna_index,
            chr_index,
            extra_categoricals,
            var_categoricals,
        ), {}

    def create_plates(
        self,
        x_data,
        idx,
        batch_index,
        label_index,
        rna_index,
        chr_index,
        extra_categoricals,
        var_categoricals,
    ):
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def list_obs_plate_vars(self):
        """
        Input dictionary for amortised infererence and minibatch sampling.

        Create a dictionary with the name of observation/minibatch plate,
        indexes of model args to provide to encoder,
        variable names that belong to the observation plate
        and the number of dimensions in non-plate axis of each variable
        """

        return {
            "name": "obs_plate",
            "input": [0, 2, 4],  # expression data + (optional) batch index
            "n_in": self.n_vars,
            "input_transform": [
                torch.log1p,
                lambda x: x,
                lambda x: x,
            ],  # how to transform input data before passing to NN
            "input_normalisation": [
                False,
                False,
                False,
            ],  # whether to normalise input data before passing to NN
            "sites": {
                "detection_y_c": 1,
                "detection_chr_y_c": 1,
                "factors_per_cell": 1,
                "cell_modules_w_cf_amortised_prior": self.n_factors,
                "cell_modules_w_cf": self.n_factors,
                "cell_type_modules_w_cz_prior": 1,
                "cell_type_modules_w_cz": self.n_labels,
            },
        }

    def get_layernorm(self, name, layer, norm_shape):

        if getattr(self.weights, f"{name}_layer_{layer}_layer_norm", None) is None:
            deep_setattr(
                self.weights,
                f"{name}_layer_{layer}_layer_norm",
                torch.nn.LayerNorm(norm_shape, elementwise_affine=False),
            )
        layer_norm = deep_getattr(self.weights, f"{name}_layer_{layer}_layer_norm")
        return layer_norm

    def get_activation(self, name, layer):

        if getattr(self.weights, f"{name}_layer_{layer}_activation_fn", None) is None:
            deep_setattr(
                self.weights,
                f"{name}_layer_{layer}_activation_fn",
                torch.nn.Softplus(),
            )
        activation_fn = deep_getattr(self.weights, f"{name}_layer_{layer}_activation_fn")
        return activation_fn

    def get_weight(self, weights_name, weights_shape):
        if not hasattr(self.weights, weights_name):
            deep_setattr(
                self.weights,
                weights_name,
                pyro.nn.PyroSample(
                    lambda prior: dist.SoftLaplace(
                        self.zeros,
                        self.ones,
                    )
                    .expand(weights_shape)
                    .to_event(len(weights_shape)),
                ),
            )
        return deep_getattr(self.weights, weights_name)

    def get_bias(self, bias_name, bias_shape):
        if not hasattr(self.weights, bias_name):
            deep_setattr(
                self.weights,
                bias_name,
                pyro.nn.PyroSample(
                    lambda prior: dist.SoftLaplace(
                        self.zeros,
                        self.ones,
                    )
                    .expand(bias_shape)
                    .to_event(len(bias_shape)),
                ),
            )
        return deep_getattr(self.weights, bias_name)

    def bayesian_fclayers(
        self,
        x: torch.Tensor,
        n_in: int,
        n_out: int,
        name="fc_genes",
    ):
        for i in range(self.n_layers):
            layer = i + 1
            if layer == self.n_layers:
                # last layer
                weights_shape = [n_out, self.n_hidden]
            elif layer == 1:
                # first layer
                bias_shape = [1, self.n_hidden]
                weights_shape = [self.n_hidden, n_in]
            else:
                # middle layers
                bias_shape = [1, self.n_hidden]
                weights_shape = [self.n_hidden, self.n_hidden]

            # optionally apply dropout ==========
            if self.dropout_rate > 0:
                if getattr(self.weights, f"{name}_layer_{layer}_dropout", None) is None:
                    deep_setattr(
                        self.weights,
                        f"{name}_layer_{layer}_dropout",
                        torch.nn.Dropout(p=self.dropout_rate),
                    )
                dropout = deep_getattr(self.weights, f"{name}_layer_{layer}_dropout")
                x = dropout(x)

            # generate parameters ==========
            weights_name = f"{name}_layer{layer}_weight"
            weights = self.get_weight(weights_name, weights_shape)
            # compute weighted sum using einsum ==========
            weights = weights / torch.tensor(float(self.n_hidden), device=weights.device).pow(0.5)
            x = torch.einsum("hf,cf->ch", weights, x)
            # optionally add bias term
            if (layer < self.n_layers) and self.decoder_bias:
                bias_name = f"{name}_layer{layer}_bias"
                bias = self.get_bias(bias_name, bias_shape)
                x = x + bias
            # apply layernorm ==========
            layer_norm = self.get_layernorm(name=name, layer=layer, norm_shape=[weights_shape[0]])
            x = layer_norm(x)
            # apply activation ==========
            activation_fn = self.get_activation(name, layer)
            x = activation_fn(x)

        return x

    def forward(
        self,
        x_data,
        idx,
        batch_index,
        label_index,
        rna_index,
        chr_index,
        extra_categoricals,
        var_categoricals,
    ):
        obs2sample = one_hot(batch_index, self.n_batch)
        if self.n_labels > 0:
            obs2label = one_hot(label_index, self.n_labels)
        obs2extra_categoricals = torch.cat(
            [
                one_hot(
                    extra_categoricals[:, i].view((extra_categoricals.shape[0], 1)),
                    n_cat,
                )
                for i, n_cat in enumerate(self.n_extra_categoricals)
            ],
            dim=1,
        )
        obs2var_categoricals = one_hot(var_categoricals, self.n_var_categoricals)
        obs_plate = self.create_plates(
            x_data,
            idx,
            batch_index,
            label_index,
            rna_index,
            chr_index,
            extra_categoricals,
            var_categoricals,
        )

        def apply_plate_to_fixed(x, index):
            if x is not None:
                return x[index]
            else:
                return x

        # =====================Cell-specific module activities ======================= #
        # module per cell activities - w_{c, f}
        with obs_plate as ind:
            w_c_dist = dist.Gamma(
                self.ones * self.factors_per_cell_alpha,
                self.factors_per_cell_beta,
            )
            factors_per_cell = pyro.sample(
                "factors_per_cell",
                w_c_dist,
            )  # (self.n_obs, 1)
            k = "cell_modules_w_cf"
            if self.use_amortised_cell_loadings_as_prior:
                k_ = k
                k = k + "_amortised_prior"
            cell_modules_w_cf = pyro.sample(
                k,
                dist.Gamma(
                    factors_per_cell / self.n_factors_torch,
                    self.ones_1_n_factors,
                ),
                obs=apply_plate_to_fixed(getattr(self, f"fixed_val_{k}", None), ind),
            )  # (self.n_obs, self.n_factors)
            if self.use_amortised_cell_loadings_as_prior:
                cell_modules_w_cf_weight = pyro.sample(
                    f"{k_}_weight",
                    dist.Gamma(
                        (self.ones + self.ones) * self.ones_1_n_factors,
                        (self.ones + self.ones),
                    ),
                    obs=apply_plate_to_fixed(getattr(self, f"fixed_val_{k_}_weight", None), ind),
                )  # (self.n_obs, self.n_factors)
                cell_modules_w_cf = pyro.deterministic(k_, cell_modules_w_cf * cell_modules_w_cf_weight)
        # cell-type module activities - w_{c, z}
        if self.n_labels > 0:
            with obs_plate as ind:
                k = "cell_type_modules_w_cz_prior"
                # defining the variable as normally
                cell_type_modules_w_cz_prior = pyro.sample(
                    k,
                    dist.Gamma(
                        factors_per_cell / self.n_factors_torch,
                        self.ones,
                    ),
                    obs=apply_plate_to_fixed(getattr(self, f"fixed_val_{k}", None), ind),
                )  # (self.n_obs, 1)
                cell_type_modules_w_cz = pyro.deterministic(
                    "cell_type_modules_w_cz", cell_type_modules_w_cz_prior * obs2label
                )  # (self.n_obs, self.n_labels)

        # =====================Cell-specific detection efficiency ======================= #
        ### RNA model ###
        if self.rna_model:
            # y_c with hierarchical mean prior
            detection_mean_y_e = pyro.sample(
                "detection_mean_y_e",
                dist.Gamma(
                    self.ones * self.detection_mean_hyp_prior_alpha,
                    self.ones * self.detection_mean_hyp_prior_beta,
                )
                .expand([self.n_batch, 1])
                .to_event(2),
            )
            beta = self.detection_hyp_prior_alpha / (obs2sample @ detection_mean_y_e)
            with obs_plate, pyro.poutine.mask(mask=rna_index):
                detection_y_c = pyro.sample(
                    "detection_y_c",
                    dist.Gamma(self.detection_hyp_prior_alpha, beta),
                )  # (self.n_obs, 1)

        ### Chromatin model ###
        if self.chromatin_model:
            # y_c with hierarchical mean prior
            detection_mean_chr_y_e = pyro.sample(
                "detection_mean_chr_y_e",
                dist.Gamma(
                    self.ones * self.detection_mean_hyp_prior_alpha,
                    self.ones * self.detection_mean_hyp_prior_beta,
                )
                .expand([self.n_batch, 1])
                .to_event(2),
            )
            beta = self.detection_hyp_prior_alpha / (obs2sample @ detection_mean_chr_y_e)
            with obs_plate, pyro.poutine.mask(mask=chr_index):
                detection_chr_y_c = pyro.sample(
                    "detection_chr_y_c",
                    dist.Gamma(self.detection_hyp_prior_alpha, beta),
                )  # (self.n_obs, 1)

        # =====================Module gene loadings ======================= #
        ### RNA model ###
        if self.rna_model and not self.use_non_linear_decoder:
            # g_{f,g}
            factor_level_g = pyro.sample(
                "factor_level_g",
                dist.Gamma(self.factor_prior_alpha, self.factor_prior_beta).expand([1, self.n_genes]).to_event(2),
                obs=getattr(self, "fixed_val_factor_level_g", None),
            )
            g_fg = pyro.sample(
                "g_fg",
                dist.Gamma(
                    self.modules_per_gene / self.n_factors_torch,
                    self.ones / factor_level_g,
                )
                .expand([self.n_factors, self.n_genes])
                .to_event(2),
                obs=getattr(self, "fixed_val_g_fg", None),
            )
            if self.n_labels > 0:
                cell_type_g_zg = pyro.sample(
                    "cell_type_g_zg",
                    dist.Gamma(
                        self.modules_per_gene / self.n_factors_torch,
                        self.ones / factor_level_g,
                    )
                    .expand([self.n_labels, self.n_genes])
                    .to_event(2),
                    obs=getattr(self, "fixed_val_cell_type_g_zg", None),
                )

        ### Chromatin model ###
        if self.chromatin_model and not self.use_non_linear_decoder:
            # g_{f,r}
            factor_level_r = pyro.sample(
                "factor_level_r",
                dist.Gamma(self.factor_prior_alpha, self.factor_prior_beta).expand([1, self.n_regions]).to_event(2),
                obs=getattr(self, "fixed_val_factor_level_r", None),
            )
            g_fr = pyro.sample(
                "g_fr",
                dist.Gamma(
                    self.modules_per_region / self.n_factors_torch,
                    self.ones / factor_level_r,
                )
                .expand([self.n_factors, self.n_regions])
                .to_event(2),
                obs=getattr(self, "fixed_val_g_fr", None),
            )
            if self.n_labels > 0:
                cell_type_g_fr = pyro.sample(
                    "cell_type_g_fr",
                    dist.Gamma(
                        self.modules_per_region / self.n_factors_torch,
                        self.ones / factor_level_r,
                    )
                    .expand([self.n_labels, self.n_regions])
                    .to_event(2),
                    obs=getattr(self, "fixed_val_cell_type_g_fr", None),
                )

        # =====================Gene-specific multiplicative component ======================= #
        ### RNA model ###
        if self.rna_model:
            # `y_{t, g}` per gene multiplicative effect that explains the difference
            # in sensitivity between genes in each technology or covariate effect
            detection_tech_gene_tg = pyro.sample(
                "detection_tech_gene_tg",
                dist.Gamma(
                    self.ones * self.gene_tech_prior_alpha,
                    self.ones * self.gene_tech_prior_beta,
                )
                .expand([np.sum(self.n_extra_categoricals), self.n_genes])
                .to_event(2),
            )

        # =====================Region-specific additive component ======================= #
        ### Chromatin model ###
        if self.chromatin_model:
            # `y_{e, r}` per region background openness (free-floating DNA)
            region_add_er = pyro.sample(
                "region_add_er",
                dist.Gamma(
                    self.ones * self.region_add_prior_alpha,
                    self.ones * self.region_add_prior_beta,
                )
                .expand([self.n_batch, self.n_regions])
                .to_event(2),
                obs=getattr(self, "fixed_val_region_add_er", None),
            )
            # `y_{r}` per region normalisation effect
            region_y_r = pyro.sample(
                "region_y_r",
                dist.Gamma(
                    self.ones_1d.expand([1, 1]) * self.ten * self.ten,
                    self.ten * self.ten,
                )
                .expand([self.n_batch, 1])
                .to_event(2),
                obs=getattr(self, "fixed_val_region_y_r", None),
            )
            region_y_er = (
                pyro.sample(
                    "region_y_er",
                    dist.Gamma(
                        self.ones_1d.expand([self.n_batch, 1]) * (self.ten * self.ten * self.ten),
                        (self.ten * self.ten * self.ten) / region_y_r,
                    )
                    .expand([self.n_batch, self.n_regions])
                    .to_event(2),
                    obs=getattr(self, "fixed_val_region_y_er", None),
                )
                .squeeze(-2)
                .T
            )

        # =====================Gene-specific additive component ======================= #
        if self.rna_model:
            # s_{e,g} accounting for background, free-floating RNA
            s_g_gene_add_alpha_hyp = pyro.sample(
                "s_g_gene_add_alpha_hyp",
                dist.Gamma(
                    self.gene_add_alpha_hyp_prior_alpha,
                    self.gene_add_alpha_hyp_prior_beta,
                )
                .expand([1, 1])
                .to_event(2),
            )
            s_g_gene_add_mean = pyro.sample(
                "s_g_gene_add_mean",
                dist.Gamma(
                    self.gene_add_mean_hyp_prior_alpha,
                    self.gene_add_mean_hyp_prior_beta,
                )
                .expand([self.n_batch, 1])
                .to_event(2),
            )  # (self.n_batch)
            s_g_gene_add_alpha_e_inv = pyro.sample(
                "s_g_gene_add_alpha_e_inv",
                dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2),
            )  # (self.n_batch)
            s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)

            s_g_gene_add = pyro.sample(
                "s_g_gene_add",
                dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
                .expand([self.n_batch, self.n_genes])
                .to_event(2),
            )  # (self.n_batch, n_genes)

        # =====================Gene-specific stochastic/unexplained variance ======================= #
        ### RNA model ###
        if self.rna_model:
            stochastic_v_ag_hyp = pyro.sample(
                "stochastic_v_ag_hyp",
                dist.Gamma(
                    self.stochastic_v_ag_hyp_prior_alpha,
                    self.stochastic_v_ag_hyp_prior_beta,
                )
                .expand([self.n_var_categoricals, 1])
                .to_event(2),
            )
            stochastic_v_ag_inv = pyro.sample(
                "stochastic_v_ag_inv",
                dist.Exponential(stochastic_v_ag_hyp)
                # dist.Exponential(
                #    self.stochastic_v_ag_hyp_prior_alpha
                #    / self.stochastic_v_ag_hyp_prior_beta
                # )
                .expand([self.n_var_categoricals, self.n_genes]).to_event(2),
            )  # (self.n_var_categoricals or 1, self.n_genes)
            stochastic_v_ag = obs2var_categoricals @ (self.ones / stochastic_v_ag_inv.pow(2))

        ### Chromatin model ###
        if self.chromatin_model and not self.use_binary_chromatin:
            stochastic_chr_v_ag_hyp = pyro.sample(
                "stochastic_chr_v_ag_hyp",
                dist.Gamma(
                    self.stochastic_v_ag_hyp_prior_alpha,
                    self.stochastic_v_ag_hyp_prior_beta,
                )
                .expand([self.n_var_categoricals, 1])
                .to_event(2),
            )
            stochastic_chr_v_ag_inv = pyro.sample(
                "stochastic_chr_v_ag_inv",
                dist.Exponential(stochastic_chr_v_ag_hyp).expand([self.n_var_categoricals, self.n_regions]).to_event(2),
            )  # (self.n_var_categoricals or 1, self.n_genes)
            stochastic_chr_v_ag = obs2var_categoricals @ (self.ones / stochastic_chr_v_ag_inv.pow(2))

        # =====================Expected expression ======================= #
        # concatenate the latent variables cell_modules_w_cf and cell_type_modules_w_cz
        if self.n_labels > 0:
            cell_modules_w_cf = torch.cat([cell_modules_w_cf, cell_type_modules_w_cz], dim=-1)

        if self.rna_model:
            ### RNA model ###
            if self.use_non_linear_decoder:
                # use a non-linear decoder to model the expected expression
                mu_biol = self.bayesian_fclayers(
                    cell_modules_w_cf,
                    n_in=self.n_factors + self.n_labels,
                    n_out=self.n_genes,
                )
            else:
                # concatenate cell_type_g_zg with g_fg
                if self.n_labels > 0:
                    g_fg = torch.cat([g_fg, cell_type_g_zg], dim=-2)
                if self.use_orthogonality_constraint:
                    zero_diag = -torch.diag(torch.tensor(1.0, device=x_data.device).expand(g_fg.shape[-2])) + self.ones
                    pyro.sample(
                        "g_fg_constraint",
                        dist.Normal(((g_fg @ g_fg.T) * zero_diag).sum(), self.ones).to_event(2),
                        obs=self.zeros,
                    )

                # per cell biological expression
                mu_biol = cell_modules_w_cf @ g_fg

            mu = (
                (mu_biol + obs2sample @ s_g_gene_add)  # contaminating RNA
                * detection_y_c
                * (obs2extra_categoricals @ detection_tech_gene_tg)
            )  # cell and gene-specific normalisation

        # =====================Expected chromatin state ======================= #
        if self.chromatin_model:
            ### Chromatin model ###
            if self.use_non_linear_decoder:
                # use a non-linear decoder to model the expected accessibility
                kon_cr = self.bayesian_fclayers(
                    cell_modules_w_cf,
                    n_in=self.n_factors + self.n_labels,
                    n_out=self.n_regions,
                    name="accessibility",
                )
            else:
                # concatenate cell_type_g_fr with g_fr
                if self.n_labels > 0:
                    g_fr = torch.cat([g_fr, cell_type_g_fr], dim=-2)
                if self.use_orthogonality_constraint:
                    zero_diag = -torch.diag(torch.tensor(1.0, device=x_data.device).expand(g_fr.shape[-2])) + self.ones
                    pyro.sample(
                        "g_fr_constraint",
                        dist.Normal(((g_fr @ g_fr.T) * zero_diag).sum(), self.ones).to_event(2),
                        obs=self.zeros,
                    )
                kon_cr = cell_modules_w_cf @ g_fr

            # per cell type and per cell rates
            # \mu_{f,r}
            kon_cr = (kon_cr + obs2sample @ region_add_er) * (obs2sample @ region_y_er)
            # per cell technical detection probabilities
            kon_cr = kon_cr * detection_chr_y_c

            if self.use_binary_chromatin:
                pon_cr = kon_cr / (kon_cr + self.ones)

        # =====================DATA likelihood ======================= #
        ### RNA model ###
        if self.rna_model:
            # Likelihood (switch / 2 state model)
            with obs_plate, pyro.poutine.mask(mask=rna_index):
                pyro.sample(
                    "data_rna",
                    dist.GammaPoisson(concentration=stochastic_v_ag, rate=stochastic_v_ag / mu),
                    obs=x_data[:, self.gene_ind_tt],
                )

        ### Chromatin model ###
        if self.chromatin_model:
            if self.use_binary_chromatin:
                # Likelihood
                with obs_plate, pyro.poutine.mask(mask=chr_index):
                    pyro.sample(
                        "data_chromatin",
                        dist.Binomial(
                            total_count=self.ones * self.binomial_n_tt,
                            probs=pon_cr,
                        ),
                        obs=x_data[:, self.region_ind_tt],
                    )
            else:
                # Likelihood
                with obs_plate, pyro.poutine.mask(mask=chr_index):
                    pyro.sample(
                        "data_chromatin",
                        dist.GammaPoisson(concentration=stochastic_chr_v_ag, rate=stochastic_chr_v_ag / kon_cr),
                        obs=x_data[:, self.region_ind_tt],
                    )

    # =====================Other functions======================= #
    def compute_expected(self, samples, adata_manager, ind_x=None):
        r"""Compute expected expression of each gene in each cell.

        Useful for evaluating how well the model learned expression
        pattern of all genes in the data.

        Parameters
        ----------
        use_pon
            use estimate probability of gene activation in every cell
            or use expected value based on k_on and k_off rates?
        """
        if ind_x is None:
            ind_x = np.arange(adata_manager.adata.n_obs).astype(int)
        else:
            ind_x = ind_x.astype(int)
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten()).values[ind_x, :]
        extra_categoricals = adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
        obs2var_categoricals = adata_manager.get_from_registry("var_categoricals")
        obs2var_categoricals = pd.get_dummies(obs2var_categoricals.flatten())
        obs2extra_categoricals = np.concatenate(
            [pd.get_dummies(extra_categoricals.iloc[ind_x, i]) for i, n_cat in enumerate(self.n_extra_categoricals)],
            axis=1,
        )

        # per cell biological expression
        mu_biol = np.dot(samples["cell_modules_w_cf"][ind_x, :], samples["g_fg"])
        if "cell_type_modules_w_cz" in samples:
            mu_biol = mu_biol + np.dot(samples["cell_type_modules_w_cz"][ind_x, :], samples["cell_type_g_zg"])
        mu = (
            (mu_biol + np.dot(obs2sample, samples["s_g_gene_add"]))  # contaminating RNA
            * samples["detection_y_c"][ind_x, :]
            * np.dot(obs2extra_categoricals, samples["detection_tech_gene_tg"])
        )  # cell and gene-specific normalisation
        # computing stochastic variance component
        stochastic_v_ag = np.dot(
            obs2var_categoricals,
            np.ones((1, 1)) / np.power(samples["stochastic_v_ag_inv"], 2),
        )

        return {"mu": mu, "alpha": stochastic_v_ag[ind_x, :]}

    def compute_expected_subset(self, samples, adata_manager, fact_ind, cell_ind):
        r"""Compute expected expression of each gene in each cell that comes from a subset of factors / cell types.

        Useful for evaluating how well the model learned expression pattern of all genes in the data.

        Parameters
        ----------
        use_pon
            use estimate probability of gene activation in every cell
            or use expected value based on k_on and k_off rates?
        """
        obs2sample = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        obs2sample = pd.get_dummies(obs2sample.flatten())
        extra_categoricals = adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
        obs2var_categoricals = adata_manager.get_from_registry("var_categoricals")
        obs2var_categoricals = pd.get_dummies(obs2var_categoricals.flatten())
        obs2extra_categoricals = np.concatenate(
            [pd.get_dummies(extra_categoricals.iloc[:, i]) for i, n_cat in enumerate(self.n_extra_categoricals)],
            axis=1,
        )

        # per cell biological expression
        mu_biol = np.dot(
            samples["cell_modules_w_cf"][cell_ind, fact_ind],
            samples["g_fg"][fact_ind, :],
        )
        if "cell_type_modules_w_cz" in samples:
            mu_biol = mu_biol + np.dot(samples["cell_type_modules_w_cz"][cell_ind, :], samples["cell_type_g_zg"])
        mu = (
            (mu_biol + np.dot(obs2sample[cell_ind, :], samples["s_g_gene_add"]))  # contaminating RNA
            * samples["detection_y_c"][cell_ind, :]
            * np.dot(obs2extra_categoricals[cell_ind, :], samples["detection_tech_gene_tg"])
        )  # cell and gene-specific normalisation
        stochastic_v_ag = np.dot(
            obs2var_categoricals,
            np.ones((1, 1)) / np.power(samples["stochastic_v_ag_inv"], 2),
        )

        return {"mu": mu, "alpha": stochastic_v_ag[cell_ind, :]}
