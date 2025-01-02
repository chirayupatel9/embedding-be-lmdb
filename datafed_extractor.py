import json
import os
import time
from datafed.CommandLib import API
import json
from flatten_dict.reducers import make_reducer
from flatten_dict import flatten
import numpy as np


def generate_sibling_context(data, parent_key=""):
    """
    Generates sibling context for each key in the input dictionary.

    Args:
        data (dict): The input dictionary.
        parent_key (str): The parent key of the current dictionary.

    Returns:
        list: A list of dictionary containing the sibling context for each key.
    """
    output_list = []
    for key, value in data.items():
        if isinstance(value, dict):
            current_dict = {parent_key + "." + key if parent_key else key: [val for key, val in data.items()
                                                                            if type(val) != dict]}
            output_list.append(current_dict)
            output_list.extend(generate_sibling_context(value, parent_key + "." + key if parent_key else key))
        else:
            current_key = parent_key + "." + key if parent_key else key
            current_dict = {current_key: [data[sibling_key] for sibling_key in data.keys() if
                                          sibling_key != key and type(data[sibling_key]) != dict]}
            output_list.append(current_dict)
    return output_list


def flatten_context(value):
    flat_v = []
    for v in value:
        if type(v) != list:
            flat_v.append(v)
        else:
            flat_v.extend((v))
    return (flat_v)


def get_sibling_cache_from_context(output):
    sibling_cache = {}
    for out in output:
        for key, value in out.items():
            flat_v = flatten_context(value)
            sibling_cache[key] = flat_v

    return (sibling_cache)


def get_prep_dict(key, value, datafed_id):
    prep_dict = {}
    prep_dict['datafed_id'] = datafed_id
    prep_dict['field_name'] = key[-1]
    prep_dict['field_value'] = (value)

    path_name = ".".join(key[0:len(key) - 1])
    prep_dict['path_name'] = path_name
    prep_dict['sibling_context'] = sibling_cache[(".".join(key[0:len(key)]))]
    path_context = []
    current = key[0]

    if path_name == "":
        pass
    else:
        if sibling_cache[key[0]] != []:
            path_context.append(sibling_cache[key[0]])
        for i in range(1, len(key) - 1):
            current += "." + key[i]
            if sibling_cache[current] != []:
                path_context.append(sibling_cache[current])

    prep_dict['path_context'] = flatten_context(path_context)

    return (prep_dict)


def get_meta(flat, datafed_id):
    prepared_list = []
    alph = {"1": "a", "2": "b", "3": "c", "4": "d", "5": "e", "6": "f", "7": "g", "8": "h", "9": "i", "10": "j",
            "11": "k", "12": "l", "13": "m", "14": "n", "15": "o", "16": "p", "17": "q", "18": "r", "19": "s",
            "20": "t", "21": "u", "22": "v", "23": "w", "24": "x", "25": "y", "26": "z"
            }
    for key, value in flat.items():
        if (type(value) == list) and value != [] and isinstance(value[0], list):
            array = np.array(value)
            rows = array.shape[0]
            cols = array.shape[1]

            for row in range(0, rows):
                for col in range(0, cols):
                    prep_dict = get_prep_dict(key, value, datafed_id)
                    new_key = str(key[-1]) + "_row" + alph[str(row + 1)] + "col" + alph[str(col + 1)]
                    prep_dict["field_name"] = new_key
                    prep_dict["field_value"] = float(array[row][col])
                    prepared_list.append(prep_dict)
        #                     print("prep dict for tensor handling", prep_dict)
        else:
            prep_dict = get_prep_dict(key, value, datafed_id)
            prepared_list.append(prep_dict)

    return (prepared_list)


df_api = API()
df_api.loginByPassword("cnp68","Chirayu#2099patel")

df_api.setContext('p/2023_symmetry_dataset_single_record')

coll_list_resp = df_api.collectionItemsList('c/463416128', count=10000)
# print(f"coll_list_resp:{coll_list_resp}")
# Ensure the directory exists
output_dir = r"static/datafed_dump"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for idx, each in enumerate(coll_list_resp[0].item):
    print(f"Processing item {each.id}")

    dv_resp = df_api.dataGet(each.id,"508555c2-96ee-11ef-94df-afe3639840c7/home/gridftp/data/globus-data/2")
    print(f'dv_resp: {idx}')
    # Check if 'metadata' exists and if it's not empty
    # if dv_resp and dv_resp[0].data and dv_resp[0].data[0].metadata:
    #     metadata_str = dv_resp[0].data[0].metadata
    #     try:
    #         res = json.loads(metadata_str)
    #     except json.JSONDecodeError as e:
    #         print(f"Error decoding JSON for item {each.id}: {e}")
    #         continue  # Skip to the next item if JSON is invalid
    #
    #     # Safely access 'formula_pretty' and 'symmetry' fields using .get()
    #     formula_pretty = res.get('formula_pretty', 'unknown')
    #     symmetry = res.get('symmetry', {})
    #     crystal_system = symmetry.get('crystal_system', 'unknown')
    #     symbol = symmetry.get('symbol', 'unknown')
    #
    #     # Handle the case when all fields are "unknown"
    #     if formula_pretty == 'unknown' and crystal_system == 'unknown' and symbol == 'unknown':
    #         print(f"Skipping item {each.id} due to missing key fields.")
    #         continue
    #
    #     # Create the filename from the metadata
    #     file_name = f"{each.id}_{formula_pretty}_{crystal_system}_{symbol}.json"
    #     file_name = file_name.replace("/", "_")
    #     string_index = file_name.find("_", 2)
    #     output_file_name = file_name[string_index + 1:]
    #     datafed_id = file_name[:string_index].replace("_", "/")
    #
    #     output = generate_sibling_context(res)
    #     sibling_cache = get_sibling_cache_from_context(output)
    #     flat = flatten(res)
    #     prepared_list = get_meta(flat, datafed_id)
    #
    #     file_path = os.path.join(output_dir, output_file_name)
    #     with open(file_path, "w") as outfile:
    #         json.dump(prepared_list, outfile)
    # else:
    #     print(f"Metadata missing or empty for item {each.id}")
