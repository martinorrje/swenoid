"""Register Swenoid tasks and invoke MjLab training."""


def main() -> None:
    from mjlab.scripts.train import main as mjlab_main

    import swenoid.tasks  # noqa: F401

    mjlab_main()
