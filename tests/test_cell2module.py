import pytest
import torch
from scvi.data import synthetic_iid

from cell2module.models import Cell2ModuleModel


@pytest.mark.parametrize("use_orthogonality_constraint", [True, False])
@pytest.mark.parametrize("amortised", [True, False])
def test_cell2location(use_orthogonality_constraint, amortised):
    save_path = "./Cell2ModuleModel_model_test"
    if torch.cuda.is_available():
        use_gpu = int(torch.cuda.is_available())
    else:
        use_gpu = False
    dataset = synthetic_iid(n_labels=5)
    Cell2ModuleModel.setup_anndata(dataset, labels_key="labels", batch_key="batch", variance_categories="batch")

    # train regression model to get signatures of cell types
    sc_model = Cell2ModuleModel(
        dataset,
        n_factors=20,
        rna_model=True,
        chromatin_model=False,
        use_orthogonality_constraint=use_orthogonality_constraint,
        amortised=amortised,
        encoder_mode="multiple",
    )
    # test full data training
    sc_model.train(max_epochs=1, use_gpu=use_gpu)
    # test minibatch training
    sc_model.train(max_epochs=1, batch_size=100, use_gpu=use_gpu)
    # export the estimated cell abundance (summary of the posterior distribution)
    factor_names_keys = {
        "g_fg": "factor_names",
        "cell_modules_w_cf": "factor_names",
        "cell_type_g_zg": "labels",
        "cell_type_modules_w_cz": "labels",
    }
    if amortised:
        add_to_varm = ["means", "stds", "q05", "q50", "q95"]
    else:
        add_to_varm = ["means", "stds", "q50"]
    dataset = sc_model.export_posterior(
        dataset,
        sample_kwargs={"batch_size": 10},
        add_to_varm=add_to_varm,
        use_quantiles=True,
        factor_names_keys=factor_names_keys,
    )
    sc_model.plot_QC(summary_name="q50")
    # TODO fix posterior sampling 0-dim array issue
    dataset = sc_model.export_posterior(
        dataset,
        sample_kwargs={"num_samples": 20, "batch_size": 50},
        use_quantiles=False,
        factor_names_keys=factor_names_keys,
    )
    # test plot_QC
    sc_model.plot_QC()

    # test save/load
    # TODO fix issue with saving anndata
    sc_model.save(save_path, overwrite=True, save_anndata=False)
    sc_model = Cell2ModuleModel.load(save_path, dataset)
    # test minibatch training
    sc_model.train(max_epochs=1, batch_size=100, use_gpu=use_gpu)
    # export the estimated cell abundance
    dataset = sc_model.export_posterior(
        dataset,
        sample_kwargs={"batch_size": 10},
        add_to_varm=add_to_varm,
        use_quantiles=True,
        factor_names_keys=factor_names_keys,
    )
    sc_model.plot_QC(summary_name="q50")
    # TODO fix posterior sampling 0-dim array issue
    dataset = sc_model.export_posterior(
        dataset,
        sample_kwargs={"num_samples": 20, "batch_size": 50},
        use_quantiles=False,
        factor_names_keys=factor_names_keys,
    )
