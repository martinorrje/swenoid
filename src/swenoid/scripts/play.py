"""Register Swenoid tasks and invoke MjLab policy visualization."""


def main() -> None:
    from mjlab.scripts.play import main as mjlab_main

    import swenoid.tasks  # noqa: F401

    mjlab_main()
