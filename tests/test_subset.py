from mousevision.config import load_subset

AREAS = ["VISp", "VISl", "VISal", "VISpm", "VISam", "VISrl"]


def test_subset_matches_the_frozen_subset():
    subset = load_subset()

    assert subset["cre_line"] == "Cux2-CreERT2"
    assert subset["imaging_depth"] == 175
    assert subset["areas"] == AREAS
    assert set(subset["containers"]) == set(AREAS)


def test_each_area_has_an_integer_container_id():
    subset = load_subset()

    for area in subset["areas"]:
        assert isinstance(subset["containers"][area], int)


def test_experiment_ids_are_unique():
    subset = load_subset()
    core = subset["core_experiments"]
    experiment_ids = list(core["natural_scenes"]) + [
        core["drifting_gratings_VISp"],
        core["locally_sparse_noise_VISp"],
    ]

    assert len(core["natural_scenes"]) == 6
    assert len(experiment_ids) == len(set(experiment_ids))
