"""Static guards shared by every public repository of the project.

Each guard is a function ``check(root, **options) -> list[Finding]`` in its own module and a sub-command of
``python -m guards <check> <repo-root>`` (run from the directory that holds this package). The package imports
only the standard library, so a free CI runner needs no install. Accepted cases are stated in the repository's
``.guards-allow`` file (see ``allow``); the private-identifier guard has no allow-list."""
