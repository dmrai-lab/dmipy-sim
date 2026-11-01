"""The CI test groups: every ``tests/**/test_*.py`` file in exactly one of ``N`` groups, balanced by measured time.

    python tools/ci_groups.py N k        # the files of group k (0-based) of N, one per line
    python tools/ci_groups.py N --check  # every file in exactly one group, none empty

The groups are the longest-processing-time assignment of the files to ``N`` bins by ``SECONDS`` (the per-file sum of
the test durations over 5 s that a CI run reported; a file not listed weighs ``DEFAULT``, the cost of the many tests
under 5 s). The assignment is a function of the file list and these numbers only, so a new file lands in a group
without anyone editing a list; re-measure by raising ``--durations-min`` on one run. A group is a fresh process,
which also bounds the memory a session of compiled programs grows to.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 8
#: seconds of the tests over 5 s of each file (path under ``tests/``), from the CI run of dmipy-sim#861
SECONDS = {
    "test_dataset_producers.py": 222,
    "geometry/test_mesh.py": 220,
    "test_adaptive_steps.py": 213,
    "replay/test_merge_field_read.py": 202,
    "test_strand_far_grid.py": 201,
    "test_field_read_interval.py": 200,
    "test_grid_field_in_walk.py": 200,
    "geometry/test_curved_cylinder.py": 196,
    "test_bundle_raw_subset.py": 171,
    "test_delivered_vectors.py": 164,
    "test_field_source_default.py": 160,
    "test_mesh_spec.py": 145,
    "test_field_declaration.py": 134,
    "test_family_rewalk.py": 129,
    "test_time_segments.py": 127,
    "geometry/test_packed_myelinated_cylinders.py": 121,
    "test_walk_cost.py": 116,
    "test_coefficient_walk.py": 115,
    "test_concomitant_oracle.py": 107,
    "geometry/test_mesh_confinement.py": 100,
    "test_replay_pose.py": 99,
    "test_wall_counts.py": 99,
    "physics/test_surface_relaxivity_t1.py": 98,
    "replay/test_consolidate_grid.py": 92,
    "replay/test_consolidate.py": 90,
    "test_walk_spool.py": 90,
    "geometry/test_sphere_union.py": 85,
    "test_rph_constructor.py": 84,
    "backends/test_backend_hook.py": 79,
    "math/test_step_certificate.py": 79,
    "test_step_law_report.py": 75,
    "test_storage_windows.py": 68,
    "test_rewindow.py": 67,
    "test_pgste_t1.py": 64,
    "test_analytic_field_grid.py": 61,
    "backends/test_void_admission.py": 61,
    "test_substrate_pack.py": 56,
    "geometry/test_mesh_tie_band.py": 55,
    "test_coil_oracle.py": 54,
    "test_strand_field.py": 53,
    "test_compile_cache.py": 52,
    "replay/test_envelope_stated.py": 51,
    "test_pose_closed_form.py": 50,
    "test_replay_at_diffusivity.py": 50,
    "test_splice_sequence.py": 49,
    "test_compartment_carry.py": 45,
    "test_curved_wall_counts.py": 43,
    "test_train_phantom.py": 40,
    "replay/test_plan_before_read.py": 40,
    "geometry/test_label_volume.py": 39,
    "test_study.py": 38,
    "test_pose_device.py": 35,
    "test_field_resolution.py": 35,
    "test_concomitant_phase.py": 35,
    "test_packed_myelin_continuous_walk.py": 35,
    "test_pose_field_per_gate.py": 34,
    "test_pose_series.py": 33,
    "test_pose_torch.py": 33,
    "test_field_record_optional.py": 33,
    "test_field_tier_stored_gate.py": 33,
    "test_replay_coefficient_space.py": 33,
    "test_bank.py": 31,
    "test_adaptive_calibration.py": 30,
    "test_spec_producers.py": 25,
    "test_compartment_ids.py": 24,
    "geometry/test_orientation_is_the_acquisition.py": 24,
    "test_substrate_request.py": 23,
    "test_substrate_frame.py": 23,
    "test_readme_phantom.py": 22,
    "test_spec_roundtrip.py": 22,
    "test_replay_at_length_scale.py": 21,
    "test_field_target_floor.py": 21,
    "test_banded_containers.py": 21,
    "test_pack_prefix.py": 21,
    "test_save_zero.py": 20,
    "test_scanner_encoding_reaches_the_phantom.py": 20,
    "spectrum/test_strands.py": 20,
    "replay/test_reference_family.py": 19,
    "test_storage_dtype_and_comp_traj.py": 19,
    "test_replay_guide.py": 19,
    "geometry/test_myelinated_cylinder_walk.py": 18,
    "spectrum/test_packs.py": 17,
    "test_brain_from_csd.py": 17,
    "test_readers_against_the_walk.py": 17,
    "test_drawn_seeds.py": 16,
    "test_susceptibility.py": 14,
    "test_acquisition_unification.py": 14,
    "test_cpmg_multiecho.py": 14,
    "test_voxel_boundary.py": 10,
    "test_field_raster_memory.py": 10,
    "geometry/test_mesh_winding.py": 10,
    "test_bore_encoding_classes_memory.py": 10,
    "test_sub_step_dispatch.py": 9,
    "test_pose_bands.py": 9,
    "math/test_step_law.py": 8,
    "test_rph_partition.py": 8,
    "test_walk_assembly_memory.py": 7,
    "test_walker_ranges.py": 6,
    "geometry/test_bounce_budget.py": 6,
    "test_drivers_take_specs.py": 6,
    "test_api_surface.py": 6,
    "geometry/test_explicit_pool.py": 6,
    "test_phantom_run_record.py": 5,
}


#: files that read one session fixture (a walk built once per process, ``tests/conftest.py``), kept in one group so
#: the walk is paid once, not once per group
TOGETHER = (
    ("test_field_read_interval.py", "replay/test_merge_field_read.py"),
    ("test_pack_prefix.py", "test_replay_coefficient_space.py", "replay/test_plan_before_read.py", "test_pose_field_per_gate.py",
     "test_train_phantom.py", "test_pose_torch.py", "test_field_record_optional.py", "test_field_tier_stored_gate.py",
     "test_pose_series.py", "test_study.py", "test_api_surface.py"),
)


def files():
    return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("test_*.py"))


def groups(n):
    weight = lambda f: SECONDS.get(f.removeprefix("tests/"), DEFAULT)
    units = {f: [f] for f in files()}
    for together in TOGETHER:
        members = ["tests/" + t for t in together]
        assert all(m in units for m in members), together
        for m in members[1:]:
            units[members[0]] += units.pop(m)
    bins = [[0.0, []] for _ in range(n)]
    for lead, unit in sorted(units.items(), key=lambda kv: (-sum(map(weight, kv[1])), kv[0])):
        b = min(bins, key=lambda b: (b[0], len(b[1])))
        b[0] += sum(map(weight, unit))
        b[1] += unit
    return [sorted(b[1]) for b in bins], [b[0] for b in bins]


def main(argv):
    n = int(argv[1])
    g, load = groups(n)
    if argv[2] == "--check":
        flat = [f for grp in g for f in grp]
        assert sorted(flat) == files() and len(set(flat)) == len(flat) and all(g), "a test file is in no group or two"
        print("groups:", ", ".join(f"{len(grp)} files ~{t:.0f} s" for grp, t in zip(g, load)))
        return 0
    print("\n".join(g[int(argv[2])]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
