import json

# Specificați calea către fișierul JSON
JSON_PATH = r"D:\study\facultate\test_cuda\Bosma22a_segmentation_slices.json"


def count_patients():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_patients = len(data)
    patients_with_lesions = sum(1 for lesions in data.values() if lesions)
    patients_without_lesions = total_patients - patients_with_lesions

    return total_patients, patients_with_lesions, patients_without_lesions


def main():
    total, with_lesions, without_lesions = count_patients()

    print(f"Total pacienți: {total}")
    print(f"Pacienți cu leziuni (cancer): {with_lesions}")
    print(f"Pacienți fără leziuni: {without_lesions}")


if __name__ == "__main__":
    main()