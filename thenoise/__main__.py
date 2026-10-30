"""Entrypoint: parse CLI, load a single model, then serve via uvicorn."""
from __future__ import annotations

from .cli import build_parser


def _serve(args) -> None:
    from .runtime import Settings, ModelPaths, Runtime
    settings = Settings(
        device=args.device, host=args.host, port=args.port,
        upscaler_dir=args.upscaler_dir, offload_device=args.offload_device,
        controlmap_dir=args.controlmap_dir,
    )

    runtime = Runtime(settings)

    # Load the model only when all three checkpoints are supplied; otherwise
    # serve model-free (upscale only). The text encoder may be a remote worker.
    text_encoder = args.text_encoder or args.text_encoder_url
    model_paths = [args.dit, args.vae, text_encoder]
    if any(model_paths):
        if not all(model_paths):
            raise SystemExit(
                "serve: to load a model you must supply --dit, --vae and "
                "--text-encoder (or --text-encoder-url) together; omit all three "
                "to run without a model"
            )
        runtime.load(
            ModelPaths(
                dit_path=args.dit,
                vae_path=args.vae,
                text_encoder_path=args.text_encoder or "",
                lora_dir=args.lora_dir,
                text_encoder_url=args.text_encoder_url,
            ),
        )

    from .api import create_app
    import uvicorn

    app = create_app(runtime)
    if runtime.available():
        print(f"thenoise serving model '{runtime.model_name}' on {settings.device}")
    else:
        print(f"thenoise serving WITHOUT a model on {settings.device} "
              f"(only upscaling and control-map extraction are available)")
    uvicorn.run(app, host=settings.host, port=settings.port)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        _serve(args)
    elif args.command == "text-encoder-serve":
        from .remote_encoder import run_encoder_server
        run_encoder_server(args)
    elif args.command == "generate":
        from .generate import run_generate
        run_generate(args)
    elif args.command == "edit":
        from .generate import run_edit
        run_edit(args)
    elif args.command == "upscale":
        from .upscale_cli import run_upscale
        run_upscale(args)
    elif args.command == "controlmap":
        from .controlmap_cli import run_controlmap
        run_controlmap(args)
    else:  # pragma: no cover - argparse requires a subcommand
        raise SystemExit(
            "choose a subcommand: serve | text-encoder-serve | generate | edit | "
            "upscale | controlmap"
        )


if __name__ == "__main__":
    main()
