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
    assert callable(atlas.stitching.stitch_ATLAS_tiles)


def test_import_stitch_atlas_tiles():
    from atlas.stitching import stitch_ATLAS_tiles

    assert callable(stitch_ATLAS_tiles)
