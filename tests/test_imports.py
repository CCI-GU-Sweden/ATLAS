def test_import_main_modules():
    import atlas
    import atlas.alignment
    import atlas.image_analysis
    import atlas.io
    import atlas.stitching

    assert atlas.alignment is not None
    assert atlas.image_analysis is not None
    assert atlas.io is not None
    assert atlas.stitching is not None
