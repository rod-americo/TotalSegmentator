import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a DICOM series directory into a NIfTI .nii.gz file.")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing one DICOM series or a .zip with the series.")
    parser.add_argument("--output", required=True, type=Path, help="Output path for the generated .nii.gz file.")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Temporary directory used when --input is a zip file.")
    parser.add_argument("--verbose", action="store_true", help="Print dicom conversion progress.")
    return parser


def _validate_args(input_path: Path, output_path: Path, tmp_dir: Path | None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_dir():
        dicom_files = list(input_path.glob("*.dcm"))
        if len(dicom_files) == 0:
            all_files = [path for path in input_path.iterdir() if path.is_file()]
            if len(all_files) == 0:
                raise ValueError(f"Input directory is empty: {input_path}")
    elif input_path.suffix.lower() != ".zip":
        raise ValueError("Input must be a directory with DICOM slices or a .zip archive.")

    if output_path.suffixes != [".nii", ".gz"]:
        raise ValueError("Output must end with .nii.gz")
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args.input, args.output, args.tmp_dir)

    from totalsegmentator.dicom_io import dcm_to_nifti

    dcm_to_nifti(
        input_path=args.input,
        output_path=args.output,
        tmp_dir=args.tmp_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

