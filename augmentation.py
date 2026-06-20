"""Image augmentation pipeline with Albumentations and OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class AugmentationPipeline:
    """Configurable image augmentation pipeline.

    Wraps Albumentations when available, falls back to OpenCV transforms.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._albumentations_available = False
        self._pipeline = None

        try:
            import albumentations as A
            self._albumentations_available = True
        except ImportError:
            pass

    def create_transforms(
        self,
        rotate_limit: int = 30,
        flip_prob: float = 0.5,
        crop_size: tuple[int, int] = (224, 224),
        color_jitter: bool = True,
        normalize: bool = True,
        custom_transforms: list | None = None,
    ) -> object:
        """Build an augmentation pipeline.

        Args:
            rotate_limit: Max rotation angle in degrees.
            flip_prob: Probability of horizontal flip.
            crop_size: Random crop target size (H, W).
            color_jitter: Enable brightness/contrast/saturation jitter.
            normalize: Normalize to ImageNet stats.
            custom_transforms: List of Albumentations transforms to append.

        Returns:
            Albumentations Compose pipeline (or None if unavailable).
        """
        if self._albumentations_available:
            import albumentations as A

            transforms_list: list = [
                A.Rotate(limit=rotate_limit, p=0.5),
                A.HorizontalFlip(p=flip_prob),
                A.RandomCrop(height=crop_size[0], width=crop_size[1], p=0.5),
            ]

            if color_jitter:
                transforms_list.extend([
                    A.RandomBrightnessContrast(p=0.3),
                    A.HueSaturationValue(p=0.3),
                ])

            if normalize:
                transforms_list.append(
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                )

            if custom_transforms:
                transforms_list.extend(custom_transforms)

            self._pipeline = A.Compose(transforms_list)
            return self._pipeline
        else:
            return None

    def augment(
        self,
        image: np.ndarray,
        num_variations: int = 1,
        pipeline: object | None = None,
    ) -> list[np.ndarray]:
        """Apply augmentation pipeline to an image.

        Args:
            image: Input BGR image.
            num_variations: Number of augmented copies to generate.
            pipeline: Optional pipeline override. Uses stored pipeline if None.

        Returns:
            List of augmented images.
        """
        pipe = pipeline or self._pipeline

        if self._albumentations_available and pipe is not None:
            import albumentations as A
            results = []
            for _ in range(num_variations):
                augmented = pipe(image=image)
                results.append(augmented["image"])
            return results

        return self._opencv_augment(image, num_variations)

    def _opencv_augment(
        self,
        image: np.ndarray,
        num_variations: int,
        rotate_limit: int = 30,
    ) -> list[np.ndarray]:
        """Fallback augmentation using only OpenCV."""
        h, w = image.shape[:2]
        rng = np.random.RandomState(self._seed)
        results = []

        for _ in range(num_variations):
            aug = image.copy()

            angle = rng.uniform(-rotate_limit, rotate_limit)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h))

            if rng.random() > 0.5:
                aug = cv2.flip(aug, 1)

            if rng.random() > 0.5:
                brightness = rng.uniform(0.7, 1.3)
                aug = np.clip(aug.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            results.append(aug)
        return results

    def augment_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.jpg",
        num_variations: int = 3,
        pipeline: object | None = None,
    ) -> int:
        """Augment all images in a directory.

        Args:
            input_dir: Source image directory.
            output_dir: Output directory for augmented images.
            pattern: Glob pattern for images.
            num_variations: Number of augmented copies per image.
            pipeline: Optional pipeline override.

        Returns:
            Total number of augmented images created.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_file in sorted(in_path.glob(pattern)):
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            stem = img_file.stem
            ext = img_file.suffix

            cv2.imwrite(str(out_path / f"{stem}_original{ext}"), img)
            count += 1

            augmented = self.augment(img, num_variations, pipeline)
            for i, aug in enumerate(augmented):
                cv2.imwrite(str(out_path / f"{stem}_aug{i}{ext}"), aug)
                count += 1

        print(f"Augmented {count} images -> {out_path}")
        return count

    def visualize_pipeline(
        self,
        image: np.ndarray,
        num_samples: int = 8,
        output_path: str | Path | None = None,
    ) -> np.ndarray:
        """Create a grid visualization of augmentation results.

        Args:
            image: Input BGR image.
            num_samples: Number of augmented samples to show.
            output_path: Save the visualization grid.

        Returns:
            Grid image (numpy array).
        """
        augmented = self.augment(image, num_samples - 1)
        augmented.insert(0, image)

        h, w = augmented[0].shape[:2]
        cols = min(4, len(augmented))
        rows = (len(augmented) + cols - 1) // cols

        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for idx, aug in enumerate(augmented):
            r, c = divmod(idx, cols)
            resized = cv2.resize(aug, (w, h))
            canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = resized

            label = "Original" if idx == 0 else f"Aug #{idx}"
            cv2.putText(canvas, label, (c * w + 5, r * h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if output_path is not None:
            cv2.imwrite(str(output_path), canvas)
            print(f"Saved visualization to {output_path}")

        return canvas

    def export_dataset(
        self,
        images: list[np.ndarray],
        labels: list[str] | None = None,
        output_dir: str | Path = "augmented_dataset",
        num_variations: int = 3,
        pipeline: object | None = None,
    ) -> Path:
        """Export augmented dataset with optional labels.

        Args:
            images: List of input images.
            labels: Optional label per image.
            output_dir: Output directory.
            num_variations: Augmented copies per image.
            pipeline: Optional pipeline override.

        Returns:
            Path to exported dataset.
        """
        out = Path(output_dir)
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        label_lines: list[str] = []
        count = 0

        for i, img in enumerate(images):
            stem = f"img{i:05d}"
            cv2.imwrite(str(img_dir / f"{stem}_orig.jpg"), img)
            count += 1

            label = labels[i] if labels and i < len(labels) else ""
            label_lines.append(f"{stem}_orig.jpg,{label}")

            augmented = self.augment(img, num_variations, pipeline)
            for j, aug in enumerate(augmented):
                fname = f"{stem}_aug{j}.jpg"
                cv2.imwrite(str(img_dir / fname), aug)
                label_lines.append(f"{fname},{label}")
                count += 1

        csv_path = out / "labels.csv"
        csv_path.write_text("\n".join(label_lines))

        print(f"Exported {count} images with labels to {out}")
        return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Image augmentation pipeline")
    parser.add_argument("source", help="Image or directory path")
    parser.add_argument("--output", default="augmented_output", help="Output path")
    parser.add_argument("--num", type=int, default=3, help="Number of variations per image")
    parser.add_argument("--visualize", action="store_true", help="Create visualization grid")
    parser.add_argument("--rotate-limit", type=int, default=30, help="Max rotation degrees")
    args = parser.parse_args()

    pipeline = AugmentationPipeline()
    pipeline.create_transforms(rotate_limit=args.rotate_limit)

    source = Path(args.source)
    if source.is_dir():
        count = pipeline.augment_directory(source, args.output, num_variations=args.num)
        print(f"Created {count} augmented images")
    else:
        img = cv2.imread(str(source))
        if img is None:
            print(f"Error: Cannot read {source}")
            return

        if args.visualize:
            vis = pipeline.visualize_pipeline(img, num_samples=args.num + 1, output_path=f"{args.output}_vis.jpg")
            print(f"Visualization saved")
        else:
            augmented = pipeline.augment(img, args.num)
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            for i, aug in enumerate(augmented):
                cv2.imwrite(str(out / f"aug_{i}.jpg"), aug)
            print(f"Saved {len(augmented)} augmented images to {out}")


if __name__ == "__main__":
    main()
