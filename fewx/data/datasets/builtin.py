import os

from .register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .fsvod import register_fsvod_instances

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2017_train_nonvoc": ("coco/train2017", "coco/new_annotations/final_split_non_voc_instances_train2017.json"),
    "coco_2017_train_voc_10_shot": ("coco/train2017", "coco/new_annotations/final_split_voc_10_shot_instances_train2017.json"),
}

_PREDEFINED_SPLITS_FSVOD = {
    "fsvod_train": ("fsvod/images", "fsvod/annotations/fsvod_train.json"),
    "fsvod_val": ("fsvod/images", "fsvod/annotations/fsvod_val.json"),
    "fsvod_test": ("fsvod/images", "fsvod/annotations/fsvod_test.json"),
}

metadata_fsvod = {}
metadata_fsvod['fsvod_train'] = {"thing_classes": ['airplane', 'airplane_drone', 'alaskan_brown_bear', 'albatross', 'alpaca', 'ambulance', 'american_bison', 'angora_cat', 'appaloosa', 'armchair', 'asiatic_black_bear', 'australian_blacksnake', 'backpack', 'bald_eagle', 'banded_gecko', 'banded_sand_snake', 'barge', 'baseball_bat', 'basket', 'basketball', 'beaker', 'bear', 'bear_cub', 'bed', 'bee', 'beetle', 'bernese_mountain_dog', 'bettong', 'bicycle', 'big_truck', 'binder', 'biscuit_(bread)', 'black-crowned_night_heron', 'black-necked_cobra', 'black_gecko', 'black_stork', 'black_vulture', 'blanket', 'blindworm_blindworm', 'blue_point_siamese', 'boa', 'boneshaker', 'book', 'booklet', 'border_collie', 'border_terrier', 'bottle', 'bovine', 'bowl', 'brambling', 'briefcase', 'brig', 'brigantine', 'broom', 'brougham', 'brush', 'brush-tailed_porcupine', 'bull', 'bulldozer', 'bumboat', 'bumper_car', 'burmese_cat', 'bus_(vehicle)', 'buzzard', 'can', 'canal_boat', 'candle', 'canister', 'canoeing', 'cape_buffalo', 'capybara', 'carabao', 'cargo_ship', 'carton', 'cat', 'cellular_telephone', 'chair', 'chameleon', 'cheviot', 'cinnamon_bear', 'cockroach', 'common_kingsnake', 'common_starling', 'computer_keyboard', 'condor', 'convertible', 'cornet', 'cotswold', 'coupe', 'cow', 'coypu', 'crow', 'cryptoprocta', 'cup', 'curassow', 'cutter', 'cutting_tool', 'dall_sheep', 'dhow', 'dirt_bike', 'dispenser', 'dog', 'domestic_llama', 'dove', 'dress_hat', 'drone', 'duck','dune_buggy', 'egyptian_cat', 'electric_locomotive', 'elephant', 'fan', 'faucet', 'fire_engine', 'fireboat', 'flamingo', 'football', 'forklift', 'frilled_lizard', 'frog', 'gift_wrap', 'gila_monster', 'giraffe', 'glider', 'gnu', 'go-kart', 'golfcart', 'goose', 'gopher', 'green_lizard', 'grizzly', 'grocery_bag', 'grouse', 'hagfish', 'hair_dryer', 'handbag', 'hat', 'hedge_sparrow', 'helmet', 'heron', 'hinny', 'hockey_stick', 'horse', 'horseless_carriage', 'hyrax', 'iceboat','icebreaker', 'ichneumon', 'indian_cobra', 'indian_mongoose', 'indian_rat_snake', 'interceptor', 'irish_terrier', 'irish_wolfhound', 'jacket', 'jeep', 'kanchil', 'kayak', 'kinkajou', 'kite', 'kitty', 'knife', 'ladle', 'lamp', 'landing_craft', 'laptop_computer', 'lawn_mower', 'lerot', 'lippizan', 'lizard', 'lovebird', 'lugger', 'magpie', 'mailboat', 'manx', 'marco_polo_sheep', 'marine_iguana', 'marker', 'merino', 'microphone', 'minicab', 'minivan', 'mole', 'moloch', 'money', 'monitor_(computer_equipment) computer_monitor', 'morgan', 'moth', 'motor_scooter', 'motorboat', 'motorcycle', 'mountain_beaver', 'mountain_bike', 'musk_ox', 'napkin', 'night_snake', 'notebook', 'orthopter', 'otter_shrew', 'otterhound', 'owl', 'packet', 'paddle', 'paintbrush', 'palomino', 'paper_towel', 'passenger_boat', 'patrol_boat', 'peacock', 'pekinese', 'pelican', 'pen', 'penguin', 'pheasant', 'pickup_truck', 'pillow', 'pine_snake', 'pink_cockatoo', 'pinto', 'pitcher_(vessel_for_liquid)', 'plate', 'plodder', 'plover', 'polar_bear', 'pole_horse', 'pony', 'pot', 'praying_mantis', "przewalski's_horse", 'pt_boat', 'punt', 'push-bike', 'pygmy_mouse', 'quarter_horse', 'racerunner', 'racket', 'rat', 'raven', 'reconnaissance_plane', 'recreational_fishing_boat', 'refrigerator', 'remote_control', 'rhodesian_ridgeback', 'road_race', 'rock_hyrax', 'roller_coaster', 'sailboard', 'sailboat', 'sandwich', 'scissors', 'scooter', 'scorpion', 'scotch_terrier', 'sealyham_terrier', 'sheep', 'skateboard', 'skidder', 'sloth_bear', 'small_boat', 'small_motorcycle', 'snail', 'snowplow', 'sofa', 'space_shuttle', 'spider', 'spoon', 'sport_utility', 'spotted_gecko', 'standard_poodle', 'standard_schnauzer', 'stanley_steamer', 'stealth_bomber', 'stealth_fighter', 'steamboat', 'steamroller', 'subcompact', 'sunglasses', 'surfboard', 'sweater', 'tabby', 'takin', 'tape_(sticky_cloth_or_paper)', 'tasmanian_devil', 'teacup', 'teapot', 'telephone', 'texas_horned_lizard', 'tiger_cat', 'tissue_paper', 'toast_(food)', 'toothbrush', 'toothpaste', 'tortoiseshell', 'towel', 'traffic_sign', 'train','tramcar', 'trawler', 'tree_lizard', 'tree_shrew', 'trolleybus', 'truck', 'umbrella', 'vacuum_cleaner', 'viscacha', 'volleyball', 'water-drop', 'water_cart', 'water_wagon', 'white_stork', 'whitetail_prairie_dog', 'wildcat', 'wineglass', 'wisent', 'wolverine', 'wombat', 'woodlouse', 'worm_lizard', 'yellow_gecko', 'zebra'] 
}
metadata_fsvod['fsvod_val'] = {"thing_classes":
['addax', 'aircraft_carrier', 'airship', 'alligator', 'anteater', 'antelope', 'armadillo', 'baby_buggy', 'badger', 'balance_car', 'barracuda', 'barrow', 'berlin', 'bezoar_goat', 'black_leopard', 'black_rhinoceros', 'boar', 'bucket', 'cabbageworm', 'camel', 'cashmere_goat', 'chimpanzee', 'civet', 'collared_peccary', 'cornetfish', 'corvette', 'dogsled', 'dragon-lion_dance', 'earwig', 'eastern_grey_squirrel', 'european_hare', 'forest_goat', 'fox_squirrel', 'genet', 'giant_armadillo', 'giant_kangaroo', 'goral', 'gorilla', 'guanaco', 'gun', 'half_track', 'hammerhead', 'helicopter', 'hippo', 'hog-nosed_skunk', 'jaguar', 'jinrikisha', 'knitting_needle', 'koala', 'lesser_kudu', 'long-tailed_porcupine', 'manatee', 'mangabey', 'medusa', 'millipede', 'mop', 'multistage_rocket', 'oxcart', 'panzer', 'piano', 'red_squirrel', 'robot', 'scraper', 'seal', 'shopping_cart', 'shrimp', 'sloth', 'small_crocodile', 'spotted_skunk', 'swing', 'sword', 'tank', 'tiger', 'toboggan', 'turtle', 'unicycle', 'urial', 'walking_stick', 'wildboar', 'yellow-throated_marten']}


metadata_fsvod['fsvod_test'] = {"thing_classes":
['JetLev-Flyer', 'amphibian', 'aoudad', 'asian_crocodile', 'autogiro', 'ax', 'bactrian_camel', 'balloon', 'bathyscaphe', 'belgian_hare', 'binturong', 'black_rabbit', 'black_squirrel', 'bow_(weapon)', 'brahman', 'canada_porcupine', 'cheetah', 'chiacoan_peccary','chimaera', 'chinese_paddlefish', 'coin', 'crab', 'crayfish', 'cruise_missile', 'deer', 'destroyer_escort', 'dumpcart', 'elasmobranch', 'elk', 'fall_cankerworm', 'fanaloka', 'fish', 'flag', 'fox', 'garden_centipede', 'gavial', 'gemsbok', 'giant_panda', 'goat', 'guard_ship', 'guitar', 'hand_truck', 'hermit_crab', 'hog', 'horse_cart', 'horseshoe_crab', 'humvee', 'ibex', 'indian_rhinoceros', 'lander', 'langur', 'lappet_caterpillar', 'lemur', 'leopard', 'lesser_panda', 'lion', 'luge', 'malayan_tapir', 'minisub', 'monkey', 'mouflon', 'mountain_goat', 'orangutan', 'pacific_walrus', 'peba', 'pedicab', 'peludo', "pere_david's_deer", 'pistol', 'pony_cart', 'pung', 'rabbit', 'raccoon', 'reconnaissance_vehicle', 'rubic_cube', 'sassaby', 'saxophone', 'sepia', 'serow', 'shark', 'shawl', 'skibob','snow_leopard', 'snowmobil', 'sow', 'spider_monkey', 'squirrel', 'suricate', 'tadpole_shrimp', 'tiglon', 'virginia_deer', 'warthog', 'whale', 'wheelchair', 'white-tailed_jackrabbit', 'white_crocodile', 'white_rabbit', 'white_rhinoceros', 'white_squirrel', 'woolly_monkey']} 

def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_FSVOD.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_fsvod_instances(
            key,
            metadata_fsvod[key],
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
