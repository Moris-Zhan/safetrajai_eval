"""Single source of the 6-class ID contract shared across detection, tracking, evaluation and Re-ID."""

CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "car",
    2: "scooter",
    3: "small_truck",
    4: "bus",
    5: "truck",
}

# MOT GT / TrackEval class IDs are 1-based.
MOT_CLASS_ID: dict[str, int] = {name: class_id + 1 for class_id, name in CLASS_NAMES.items()}
