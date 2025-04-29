def normalize_angle(angle):
    """
    Normalizes an angle to the range [-180, 180] degrees.

    Parameters:
    ----------
    angle : float
        The input angle in degrees.

    Returns:
    -------
    float
        The normalized angle within [-180, 180] range.
    """
    return ((angle + 180) % 360) - 180