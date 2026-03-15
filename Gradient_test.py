"""Run explainability checks on a trained ISS docking regression model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from explanibility_functions import (
    get_integrated_gradients,
    load_docking_model,
    make_attribution_map,
    plot_explanations,
    predict_outputs,
    preprocess_image,
)


DEFAULT_OUTPUT_DIR = Path("explainability_outputs")
DEFAULT_IMAGE_DIR = Path("data/train")
DEFAULT_TEST_SPLIT = Path("data/test_split.csv")
DEFAULT_TEST_IMAGE_IDS = (14, 291, 8439, 6161, 5589)
OUTPUT_NAMES = {
    0: "x",
    1: "y",
    2: "distance",
}
DEFAULT_OUTPUT_INDEXES = (2,)
COMMON_MODEL_PATHS = (
    Path("models/resnet_docking_best.h5"),
    Path("models/resnet_docking_best.keras"),
    Path("outputs/resnet_docking_best.h5"),
    Path("outputs/resnet_docking_best.keras"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test explainability utilities on a trained ISS docking model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a saved Keras model (.h5 or .keras). If omitted, common paths are tried.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Explicit image paths to explain. Overrides --num-images and CSV-based selection.",
    )
    parser.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEST_IMAGE_IDS),
        help=(
            "Image numeric IDs (without .jpg) to explain when --images is not passed. "
            "Defaults to 14 291 8439 6161 5589."
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        choices=(3, 4, 5),
        help="Number of test images to explain when auto-selecting from the test split.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing image files referenced by the test split.",
    )
    parser.add_argument(
        "--test-split",
        type=Path,
        default=DEFAULT_TEST_SPLIT,
        help="CSV file with a filename column used for automatic image selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where explanation plots will be written.",
    )
    parser.add_argument(
        "--use-imagenet-norm",
        action="store_true",
        help="Apply ImageNet normalization during preprocessing.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        help="Number of interpolation steps for Integrated Gradients.",
    )
    parser.add_argument(
        "--output-indexes",
        type=int,
        nargs="+",
        default=list(DEFAULT_OUTPUT_INDEXES),
        choices=(0, 1, 2),
        help=(
            "Model output index(es) to explain: 0=x, 1=y, 2=distance. "
            "Default is only 2 (distance)."
        ),
    )
    return parser.parse_args()


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"Model file not found: {model_path}")

    for candidate in COMMON_MODEL_PATHS:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No saved Keras model file was found. Pass --model-path with a .h5 or .keras file."
    )


def select_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.images:
        image_paths = [Path(image_path) for image_path in args.images]
    elif args.image_ids:
        image_paths = [args.image_dir / f"{image_id}.jpg" for image_id in args.image_ids]
    else:
        if not args.test_split.exists():
            raise FileNotFoundError(f"Test split CSV not found: {args.test_split}")

        split_df = pd.read_csv(args.test_split)
        if "filename" not in split_df.columns:
            raise ValueError(f"Missing 'filename' column in {args.test_split}")

        image_paths = [args.image_dir / filename for filename in split_df["filename"].head(args.num_images)]

    if not image_paths:
        raise ValueError("No images were selected for explainability testing.")

    missing_paths = [str(path) for path in image_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(missing_paths)
        raise FileNotFoundError(f"Some image files were not found:\n{missing_text}")

    return image_paths


def sanitize_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in path.stem)


def print_prediction(image_name: str, prediction: list[float]) -> None:
    print(f"\nImage: {image_name}")
    print(f"  x: {prediction[0]:.6f}")
    print(f"  y: {prediction[1]:.6f}")
    print(f"  distance: {prediction[2]:.6f}")


def explain_image(
    model,
    image_path: Path,
    output_dir: Path,
    use_imagenet_norm: bool,
    show_plots: bool,
    ig_steps: int,
    output_indexes: list[int],
) -> None:
    img_batch = preprocess_image(
        str(image_path),
        use_imagenet_norm=use_imagenet_norm,
    )
    prediction = predict_outputs(model, img_batch)[0].tolist()
    print_prediction(image_path.name, prediction)

    title = image_path.name
    image_stem = sanitize_stem(image_path)

    for output_index in output_indexes:
        attributions = get_integrated_gradients(
            model,
            img_batch,
            output_index=output_index,
            num_steps=ig_steps,
        )
        heatmap = make_attribution_map(attributions)
        save_path = output_dir / f"{image_stem}_output_{output_index}_{OUTPUT_NAMES[output_index]}.png"
        plot_explanations(
            img_batch,
            ig_heatmap=heatmap,
            title=title,
            save_path=str(save_path),
            show_plot=show_plots,
        )
        print(
            f"  Saved output {output_index} ({OUTPUT_NAMES[output_index]}) explanation to: {save_path}"
        )


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    image_paths = select_image_paths(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = load_docking_model(str(model_path))
    print(f"Saving explanation plots to: {args.output_dir.resolve()}")
    print(f"Using {len(image_paths)} image(s) for explainability testing")
    print(f"Explaining output index(es): {args.output_indexes}")

    for image_path in image_paths:
        explain_image(
            model=model,
            image_path=image_path,
            output_dir=args.output_dir,
            use_imagenet_norm=args.use_imagenet_norm,
            show_plots=args.show_plots,
            ig_steps=args.ig_steps,
            output_indexes=args.output_indexes,
        )

    print("\nExplainability test run complete.")


if __name__ == "__main__":
    main()