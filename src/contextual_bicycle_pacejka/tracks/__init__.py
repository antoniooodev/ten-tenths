"""Track preparation: TUM centreline reference, minimum-curvature racing line, probed speed profile, kink repair.

Chain per circuit (files in ``data/tracks/<circuit>/``):

    tum_centerline.csv          reference.build_reference        -> reference.npz
    reference.npz               raceline.build_racing_line       -> minimum_curvature_line.npz
    minimum_curvature_line.npz  speed_profile.fit_speed_profile  -> speed_profile_probed.npz
    speed_profile_probed.npz    speed_profile.scale_profile      -> racing_line_R0.npz  (x 0.9; Monza x 0.83)
    racing_line_R0.npz          repair.repair_racing_line        -> racing_line_R1.npz  (Monza, Spa, Catalunya)

Monza differs in the second step: its line (``time_optimal_line.npz``) comes from an earlier time-optimal solver
that is no longer part of the code base and is kept as data. Budapest has no curvature kink, so no R1.
"""
