import logging
import warnings
from typing import List, Optional, Union

import numpy as np
from anndata import AnnData
from cell2location.models.base._pyro_mixin import (
    PltExportMixin,
    PyroAggressiveConvergence,
    PyroAggressiveTrainingPlan,
    PyroTrainingPlan,
    QuantileMixin,
)
from pyro import clear_param_store
from pyro.infer.autoguide import AutoHierarchicalNormalMessenger
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import DataSplitter, DeviceBackedDataSplitter
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.model.base._pyromixin import PyroJitGuideWarmup
from scvi.train import TrainRunner
from scvi.utils import setup_anndata_dsp

from cell2module.models._base_module import (
    RegressionBaseModule,
    compute_cluster_summary,
)
from cell2module.models._cell2module_module import MultimodalNMFPyroModel

logger = logging.getLogger(__name__)


class Cell2ModuleModel(
    QuantileMixin,
    PyroSampleMixin,
    PyroSviTrainMixin,
    PltExportMixin,
    BaseModelClass,
):
    """
    Regulatory programme model.

    User-end model class.

    Parameters
    ----------
    adata
        single-cell AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    use_gpu
        Use the GPU?
    **model_kwargs
        Keyword args for :class:`~scvi.external.LocationModelLinearDependentWMultiExperimentModel`
    variance_categories
        Categories expected to have differing stochastic/unexplained variance

    Examples
    --------
    TODO add example
    >>>
    """

    use_max_count_cluster_as_initial = False

    def __init__(
        self,
        adata: AnnData,
        model_class=None,
        n_factors: int = 200,
        factor_names: Optional[list] = None,
        **model_kwargs,
    ):
        # in case any other model was created before that shares the same parameter names.
        clear_param_store()

        super().__init__(adata)

        self.mi_ = []
        self.minibatch_genes_ = False

        if model_class is None:
            model_class = MultimodalNMFPyroModel

        # create factor names
        self.n_factors_ = n_factors
        if factor_names is None:
            if "n_labels" in self.summary_stats:  # and self.summary_stats["n_labels"] > 0:
                self.factor_names_ = {
                    "factor_names": np.array([f"factor_{i}" for i in range(n_factors)]),
                    "labels": self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping,
                }
                model_kwargs["n_labels"] = self.summary_stats["n_labels"]
            else:
                self.factor_names_ = np.array([f"factor_{i}" for i in range(n_factors)])
        else:
            self.factor_names_ = factor_names

        if "create_autoguide_kwargs" not in model_kwargs.keys():
            model_kwargs["create_autoguide_kwargs"] = dict()
        if "guide_class" not in model_kwargs["create_autoguide_kwargs"].keys():
            model_kwargs["create_autoguide_kwargs"]["guide_class"] = AutoHierarchicalNormalMessenger

        # annotations for extra categorical covariates
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            self.extra_categoricals_ = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            self.n_extra_categoricals_ = self.extra_categoricals_.n_cats_per_key
            model_kwargs["n_extra_categoricals"] = self.n_extra_categoricals_
        # annotations for covariates that affect stochastic variance
        if "var_categoricals" in self.adata_manager.data_registry:
            model_kwargs["n_var_categoricals"] = len(
                self.adata_manager.get_state_registry("var_categoricals").categorical_mapping
            )
        model_kwargs["n_obs"] = self.summary_stats["n_cells"]
        model_kwargs["n_vars"] = self.summary_stats["n_vars"]
        model_kwargs["n_batch"] = self.summary_stats["n_batch"]
        model_kwargs["gene_bool"] = self.adata_manager.adata.var["_gene_bool"].values

        self.module = RegressionBaseModule(
            model=model_class,
            on_load_kwargs={
                "batch_size": 50,
                "max_epochs": 1,
            },
            n_factors=self.n_factors_,
            **model_kwargs,
        )

        self._model_summary_string = f"cell2state model with the following params: \nn_factors: {self.n_factors_}"
        self.obs_names_ = self.adata_manager.adata.obs_names.values
        self.var_names_ = self.adata_manager.adata.var_names.values
        self.gene_bool_ = self.module.model.gene_bool.flatten()
        self.batch_categoricals_ = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
        self.var_categoricals_ = self.adata_manager.get_state_registry("var_categoricals").categorical_mapping
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        rna_index: Optional[str] = None,
        chr_index: Optional[str] = None,
        variance_categories: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        gene_bool_key: Optional[str] = None,
        offset_key: Optional[str] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(param_copy)s
        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        # add index for each cell (provided to pyro plate for correct minibatching)
        if "_indices" in adata.obs.columns:
            del adata.obs["_indices"]
        adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")

        # if index for cells that have RNA measurements does not exist assume all cells have RNA
        if rna_index is None:
            adata.obs["_rna_index"] = True
            rna_index = "_rna_index"
        # if index for cells that have chromatin measurements does not exist assume all cells have chromatin
        if chr_index is None:
            adata.obs["_chr_index"] = True
            chr_index = "_chr_index"

        if categorical_covariate_keys is None:
            adata.obs["_categorical_covariate"] = True
            categorical_covariate_keys = ["_categorical_covariate"]

        if gene_bool_key is None:
            adata.var["_gene_bool"] = True
        else:
            adata.var["_gene_bool"] = adata.var[gene_bool_key].values

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
            NumericalObsField("rna_index", rna_index),
            NumericalObsField("chr_index", chr_index),
        ]

        # annotations for covariates that affect stochastic variance
        if variance_categories is not None:
            anndata_fields.append(CategoricalObsField("var_categoricals", variance_categories))
        if labels_key is not None:
            anndata_fields.append(CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key))
        if continuous_covariate_keys is not None:
            anndata_fields.append(NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys))
        if offset_key is not None:
            anndata_fields.append(NumericalObsField("offset", offset_key))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: Optional[int] = 1000,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 1,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = False,
        lr: Optional[float] = None,
        plan_kwargs: Optional[dict] = None,
        use_aggressive_training: bool = True,
        ignore_warnings: bool = False,
        scale_elbo: float = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training. If `None`, no minibatching occurs and all
            data is copied to device (e.g., GPU).
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        lr
            Optimiser learning rate (default optimiser is :class:`~pyro.optim.ClippedAdam`).
            Specifying optimiser via plan_kwargs overrides this choice of lr.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        if max_epochs is None:
            n_obs = self.adata_manager.adata.n_obs
            max_epochs = np.min([round((20000 / n_obs) * 1000), 1000])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        if lr is not None and "optim" not in plan_kwargs.keys():
            plan_kwargs.update({"optim_kwargs": {"lr": lr}})
        if scale_elbo != 1.0:
            if scale_elbo is None:
                scale_elbo = 1.0 / (self.summary_stats["n_cells"] * self.summary_stats["n_vars"])
            plan_kwargs["scale_elbo"] = scale_elbo

        if batch_size is None:
            # use data splitter which moves data to GPU once
            data_splitter = DeviceBackedDataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        else:
            data_splitter = DataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
            )
        if use_aggressive_training:
            aggressive_vars = list(self.module.list_obs_plate_vars["sites"].keys()) + ["cell_modules_w_cf_weight"]
            aggressive_vars = aggressive_vars + [f"{i}_initial" for i in aggressive_vars]
            aggressive_vars = aggressive_vars + [f"{i}_unconstrained" for i in aggressive_vars]
            plan_kwargs["aggressive_vars"] = aggressive_vars
            training_plan = PyroAggressiveTrainingPlan(pyro_module=self.module, **plan_kwargs)
        else:
            training_plan = PyroTrainingPlan(pyro_module=self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]

        if "callbacks" not in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] = []
        trainer_kwargs["callbacks"].append(PyroJitGuideWarmup())
        if use_aggressive_training:
            trainer_kwargs["callbacks"].append(PyroAggressiveConvergence(patience=100, tolerance=1e-6))

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter("ignore")
            res = runner()
        if use_aggressive_training:
            self.mi_ = self.mi_ + training_plan.mi
        return res

    def _compute_cluster_summary(self, key=REGISTRY_KEYS.LABELS_KEY, summary="mean"):
        """
        Compute average per cluster (key=REGISTRY_KEYS.LABELS_KEY) or per batch (key=REGISTRY_KEYS.BATCH_KEY).

        Returns
        -------
        pd.DataFrame with variables in rows and labels in columns
        """
        # find cell label column
        label_col = self.adata_manager.get_state_registry(key).original_key

        # find data slot
        x_dict = self.adata_manager.data_registry["X"]
        if x_dict["attr_name"] == "X":
            use_raw = False
        else:
            use_raw = True
        if x_dict["attr_name"] == "layers":
            layer = x_dict["attr_key"]
        else:
            layer = None

        # compute mean expression of each gene in each cluster/batch
        aver = compute_cluster_summary(
            self.adata_manager.adata,
            labels=label_col,
            use_raw=use_raw,
            layer=layer,
            summary=summary,
        )

        return aver

    def export_posterior(
        self,
        adata,
        sample_kwargs: Optional[dict] = None,
        export_slot: str = "mod",
        export_varm_variables: list = ["g_fg", "cell_type_g_zg"],
        export_obsm_variables: list = ["cell_modules_w_cf", "cell_type_modules_w_cz"],
        add_to_varm: list = ["means", "stds", "q05", "q95"],
        add_to_obsm: Optional[list] = None,
        factor_names_keys: Optional[dict] = None,
        use_quantiles: bool = True,
    ):
        """
        Summarise posterior distribution and export results to anndata object:

        1. adata.obsm: Selected variables as pd.DataFrames for each posterior distribution summary `add_to_varm`,
            posterior mean, sd, 5% and 95% quantiles (['means', 'stds', 'q05', 'q95']).
            If export to adata.varm fails with error, results are saved to adata.var instead.
        2. adata.uns: Posterior of all parameters, model name, date,
            cell type names ('factor_names'), obs and var names.

        Parameters
        ----------
        adata
            anndata object where results should be saved
        sample_kwargs
            arguments for self.sample_posterior (generating and summarising posterior samples), namely:
                num_samples - number of samples to use (Default = 1000).
                batch_size - data batch size (keep low enough to fit on GPU, default 2048).
                use_gpu - use gpu for generating samples?
        export_slot
            adata.uns slot where to export results
        export_varm_variables
            variable/site names to export in varm
        export_obsm_variables
            variable/site names to export in obsm
        add_to_varm
            posterior distribution summary to export in adata.varm (['means', 'stds', 'q05', 'q95']).
        add_to_obsm
            posterior distribution summary to export in adata.obsm (['means', 'stds', 'q05', 'q95']).
        factor_names_keys
            if multiple factor names are present in `model.factor_names_`
            - a dictionary defining the correspondence between `export_varm_variables`/`export_obsm_variables` and
            `model.factor_names_` must be provided
        use_quantiles
            compute quantiles directly (True, more memory efficient) or use samples (False, default).
            If True, means and stds cannot be computed so are not exported and returned.
        """
        sample_kwargs = sample_kwargs if isinstance(sample_kwargs, dict) else dict()
        factor_names_keys = factor_names_keys if isinstance(factor_names_keys, dict) else dict()

        # get posterior distribution summary
        if use_quantiles:
            add_to_varm = [i for i in add_to_varm if (i not in ["means", "stds"]) and ("q" in i)]
            if len(add_to_varm) == 0:
                raise ValueError("No quantiles to export - please add add_to_varm=['q05', 'q50', 'q95'].")
            if add_to_obsm is None:
                add_to_obsm = add_to_varm
            self.samples = dict()
            for i in add_to_varm:
                q = float(f"0.{i[1:]}")
                self.samples[f"post_sample_{i}"] = self.posterior_quantile(q=q, use_median=True, **sample_kwargs)
        else:
            if add_to_obsm is None:
                add_to_obsm = add_to_varm
            # generate samples from posterior distributions for all parameters
            # and compute mean, 5%/95% quantiles and standard deviation
            self.samples = self.sample_posterior(**sample_kwargs)

        # export posterior distribution summary for all parameters and
        # annotation (model, date, var, obs and cell type names) to anndata object
        adata.uns[export_slot] = self._export2adata(self.samples)

        # export estimated gene loadings
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix

        for var in export_varm_variables:
            for k in add_to_varm:
                if type(self.factor_names_) is dict:
                    factor_names_key = factor_names_keys[var]
                else:
                    factor_names_key = ""
                sample_df = self.sample2df_vars(
                    self.samples,
                    site_name=var,
                    summary_name=k,
                    name_prefix="",
                    factor_names_key=factor_names_key,
                )
                try:
                    adata.varm[f"{k}_{var}"] = sample_df.loc[adata.var.index, :]
                except ValueError:
                    # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                    adata.var[sample_df.columns] = sample_df.loc[adata.var.index, :]

        # add estimated per cell regulatory programme activities as dataframe to obsm in anndata
        # data frames contain mean, 5%/95% quantiles and standard deviation, denoted by a prefix
        for var in export_obsm_variables:
            for k in add_to_obsm:
                if type(self.factor_names_) is dict:
                    factor_names_key = factor_names_keys[var]
                else:
                    factor_names_key = ""
                sample_df = self.sample2df_obs(
                    self.samples,
                    site_name=var,
                    summary_name=k,
                    name_prefix="",
                    factor_names_key=factor_names_key,
                )
                try:
                    adata.obsm[f"{k}_{var}"] = sample_df.loc[adata.obs.index, :]
                except ValueError:
                    # Catching weird error with obsm: `ValueError: value.index does not match parent’s axis 1 names`
                    adata.obs[sample_df.columns] = sample_df.loc[adata.obs.index, :]

        return adata
