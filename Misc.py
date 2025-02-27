import os
import json


def filter_non_existing_patients(gleason_json_path, output_json_path, new_json_path):
    """
    Încarcă datele din 'gleason_json_path' (de exemplu output_mha_t2w.json) și
    din 'output_json_path'. Creează un nou JSON care conține doar pacienții care sunt
    prezenți în gleason_json_path, dar nu se regăsesc în output_json_path.

    Args:
        gleason_json_path (str): Calea către fișierul JSON cu informațiile de tip GS și datele RMN (ex: output_mha_t2w.json).
        output_json_path (str): Calea către fișierul JSON output.json.
        new_json_path (str): Calea la care se va salva noul JSON.
    """
    # Încarcă fișierele JSON
    with open(gleason_json_path, 'r') as f:
        gleason_data = json.load(f)

    with open(output_json_path, 'r') as f:
        output_data = json.load(f)

    # Cheile din output.json
    output_keys = set(output_data.keys())

    # Construiește un nou dicționar cu pacienții care nu se regăsesc în output.json
    new_data = {patient: info for patient, info in gleason_data.items() if patient not in output_keys}

    # Salvează rezultatul
    os.makedirs(os.path.dirname(new_json_path), exist_ok=True)
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    print(f"New JSON saved to {new_json_path} with {len(new_data)} patients.")


# Exemplu de utilizare:
if __name__ == "__main__":
    gleason_json_path = r"D:\study\facultate\test_cuda\data\output_mha_t2w.json"
    output_json_path = r"D:\study\facultate\test_cuda\data\output.json"
    new_json_path = r"D:\study\facultate\test_cuda\data\non_existing_patients.json"
    filter_non_existing_patients(gleason_json_path, output_json_path, new_json_path)
