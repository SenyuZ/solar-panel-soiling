from solarsoil import taxonomy


def test_label_spaces():
    assert taxonomy.label_space("binary") == ["clean", "dirty"]
    assert len(taxonomy.label_space("multiclass")) == 6
    assert taxonomy.label_space("condition") == ["clean", "soiled", "damaged"]


def test_normalize_label():
    assert taxonomy.normalize_label("clean") == "Clean"
    assert taxonomy.normalize_label("bird_drop") == "Bird-drop"
    assert taxonomy.normalize_label("Snow") == "Snow-Covered"


def test_to_label_space():
    assert taxonomy.to_label_space("Dusty", "binary") == "dirty"
    assert taxonomy.to_label_space("Clean", "binary") == "clean"
    assert taxonomy.to_label_space("Dusty", "condition") == "soiled"
    assert taxonomy.to_label_space("Physical-Damage", "condition") == "damaged"
    assert taxonomy.to_label_space("Bird-drop", "multiclass") == "Bird-drop"
