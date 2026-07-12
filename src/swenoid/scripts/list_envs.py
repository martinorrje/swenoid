"""List MjLab environments after registering Swenoid tasks."""


def main() -> None:
    from mjlab.scripts.list_envs import main as mjlab_main

    import swenoid.tasks  # noqa: F401

    mjlab_main()
