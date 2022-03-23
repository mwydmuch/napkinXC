Python API
==========

.. currentmodule:: napkinxc

Models
------

.. autosummary::
    :toctree: python_api/

    models.PLT
    models.HSM
    models.BR
    models.OVR

Datasets
--------

.. autosummary::
    :toctree: python_api/

    datasets.download_dataset
    datasets.load_dataset
    datasets.load_libsvm_file
    datasets.load_json_lines_file
    datasets.to_csr_matrix
    datasets.to_np_matrix

Measures
--------

.. autosummary::
    :toctree: python_api/

    measures.precision_at_k
    measures.recall_at_k
    measures.coverage_at_k
    measures.dcg_at_k
    measures.Jain_et_al_inverse_propensity
    measures.Jain_et_al_propensity
    measures.ndcg_at_k
    measures.psprecision_at_k
    measures.psrecall_at_k
    measures.psdcg_at_k
    measures.psndcg_at_k
    measures.hamming_loss
    measures.f1_measure