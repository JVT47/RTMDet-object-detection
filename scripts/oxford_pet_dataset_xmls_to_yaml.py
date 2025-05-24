"""
Script used to convert xml annotations for the Oxford-IIIT Pet Dataset into train, test, and valid yaml files.

Example usage:
- uv run -m scripts.oxford_pet_dataset_xmls_to_yaml --xmls-dir data/annotations/xmls
    --save-dir-path data/annotations
"""

from argparse import ArgumentParser
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import yaml


BREAD_IDS = {
    "abyssinian": 0,
    "american_bulldog": 1,
    "american_pit_bull_terrier": 2,
    "basset_hound": 3,
    "beagle": 4,
    "bengal": 5,
    "birman": 6,
    "bombay": 7,
    "boxer": 8,
    "british_shorthair": 9,
    "chihuahua": 10,
    "egyptian_mau": 11,
    "english_cocker_spaniel": 12,
    "english_setter": 13,
    "german_shorthaired": 14,
    "great_pyrenees": 15,
    "havanese": 16,
    "japanese_chin": 17,
    "keeshond": 18,
    "leonberger": 19,
    "maine_coon": 20,
    "miniature_pinscher": 21,
    "newfoundland": 22,
    "persian": 23,
    "pomeranian": 24,
    "pug": 25,
    "ragdoll": 26,
    "russian_blue": 27,
    "saint_bernard": 28,
    "samoyed": 29,
    "scottish_terrier": 30,
    "shiba_inu": 31,
    "siamese": 32,
    "sphynx": 33,
    "staffordshire_bull_terrier": 34,
    "wheaten_terrier": 35,
    "yorkshire_terrier": 36,
}


def xml_to_dict(element: Element) -> dict | str:
    result = {}

    for child in element:
        child_data = xml_to_dict(child)
        if child.tag == "object":
            result[child.tag] = result.get(child.tag, []) + [child_data]
        else:
            result[child.tag] = child_data

    text = element.text.strip() if element.text else ""
    if text:
        return text

    return result


def get_breed_from_filename(filename: str) -> str:
    pattern = r"^(.*?)_\d+\.jpg$"

    match = re.match(pattern, filename)

    if match is None:
        raise RuntimeError(
            f"Could not match pattern: {pattern} to filename: {filename}"
        )
    return match.group(1).lower()


xml_dir = Path("data", "annotations", "xmls")


def clean_annotation_dict(annotation: dict) -> dict:
    """
    Returns a dict with the interesting parts for this project.
    """

    def clean_object_dict(filename: str, object_dict: dict) -> dict:
        result = {}
        result["species"] = object_dict["name"]
        result["breed"] = get_breed_from_filename(filename)
        result["breed_id"] = BREAD_IDS[result["breed"]]
        result["bbox"] = {k: float(v) for k, v in object_dict["bndbox"].items()}
        return result

    result = {}
    filename = annotation["filename"]
    result["filename"] = filename
    result["image"] = {
        "width": float(annotation["size"]["width"]),
        "height": float(annotation["size"]["height"]),
    }
    result["objects"] = [
        clean_object_dict(filename, object_dict) for object_dict in annotation["object"]
    ]

    return result


def save_to_yaml(save_file_path: Path, annotations: list[dict]) -> None:
    yaml_dict = {"annotations": annotations}
    with open(save_file_path, "w") as f:
        yaml.dump(data=yaml_dict, stream=f, default_flow_style=False)


def xmls_to_yaml(
    xmls_path: Path, save_dir_path: Path, test_portion: float, valid_portion: float
) -> None:
    annotations = []
    for xml_file in xmls_path.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        annotation = xml_to_dict(root)

        if isinstance(annotation, str):
            raise ValueError(f"Annotation '{annotation}' should be dict. Received str")

        annotations.append(clean_annotation_dict(annotation))

    random.shuffle(annotations)
    n_annotations = len(annotations)

    train_upper_index = int(test_portion * n_annotations)
    valid_upper_index = int((test_portion + valid_portion) * n_annotations)

    save_to_yaml(save_dir_path.joinpath("train.yaml"), annotations[:train_upper_index])
    save_to_yaml(
        save_dir_path.joinpath("valid.yaml"),
        annotations[train_upper_index:valid_upper_index],
    )
    save_to_yaml(save_dir_path.joinpath("test.yaml"), annotations[valid_upper_index:])


def main() -> None:
    arg_parser = ArgumentParser(
        description="Convert xml annotations for the Oxford-IIIT Pet Dataset into a single yaml file"
    )
    arg_parser.add_argument(
        "--xmls-dir", required=True, help="Path to the dir with the xml annotations"
    )
    arg_parser.add_argument(
        "--save-dir-path",
        required=True,
        help="Dir path where the train.yaml, test.yaml, and valid.yaml are saved",
    )
    arg_parser.add_argument(
        "--train-portion",
        default=0.7,
        type=float,
        help="Portion of data used for training data.",
    )
    arg_parser.add_argument(
        "--valid-portion",
        default=0.2,
        type=float,
        help="Portion of data used for validation data.",
    )
    arg_parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used to initialize the random number generator. Different seed produces different train, test, and valid split",
    )

    args = arg_parser.parse_args()

    xmls_path = Path(args.xmls_dir)
    save_dir_path = Path(args.save_dir_path)
    train_portion = args.train_portion
    valid_portion = args.valid_portion

    if (train_portion < 0 or valid_portion < 0) or train_portion + valid_portion > 1:
        raise ValueError(
            "train-portion and valid-portion must be nonnegative and their sum must be at most 1."
        )

    random.seed(args.seed)

    xmls_to_yaml(xmls_path, save_dir_path, train_portion, valid_portion)


if __name__ == "__main__":
    main()
