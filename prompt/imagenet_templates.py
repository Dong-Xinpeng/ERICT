
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


# imagenet-A

imagenet_A_class_name_full = ['bullfrog, Rana catesbeiana', 'ambulance', 'school bus', 'lorikeet', 'African chameleon, Chamaeleo chamaeleon', 'golfcart, golf cart', 'chain', 
                              'lion, king of beasts, Panthera leo', 'organ, pipe organ', 'ocarina, sweet potato', 'Christmas stocking', 'African elephant, Loxodonta africana', 
                              'volleyball', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 
                              'vulture', 'nail', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'toucan', 'common iguana, iguana, Iguana iguana', 'go-kart', 
                              "academic gown, academic robe, judge's robe", 'goblet', 'golden retriever', 'umbrella', 'drake', 'wreck', 'volcano', 'iron, smoothing iron', 
                              'goldfinch, Carduelis carduelis', 'sea lion', 'leafhopper', 'syringe', 'hummingbird', 'puck, hockey puck', 'skunk, polecat, wood pussy', 
                              'red fox, Vulpes vulpes', 'barn', 'candle, taper, wax light', "jack-o'-lantern", 'pool table, billiard table, snooker table', 'kimono', 
                              'cucumber, cuke', 'jellyfish', 'scorpion', 'banjo', 'Persian cat', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 
                              'beacon, lighthouse, beacon light, pharos', 'carbonara', 'obelisk', 'hotdog, hot dog, red hot', 'garter snake, grass snake', 
                              'robin, American robin, Turdus migratorius', 'mushroom', 'corn', 'lighter, light, igniter, ignitor', 'American egret, great white heron, Egretta albus', 
                              'fountain', 'piggy bank, penny bank', 'stingray', 'wine bottle', 'crayfish, crawfish, crawdad, crawdaddy', 'garbage truck, dustcart', 'acoustic guitar', 
                              'balance beam, beam', 'apron', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'grand piano, grand', 'broccoli', 'soap dispenser', 
                              'dial telephone, dial phone', 'limousine, limo', 'fly', 'rocking chair, rocker', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 
                              'lycaenid, lycaenid butterfly', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'lynx, catamount', 'hermit crab', 'bison', 'mask', 'Rottweiler', 
                              'submarine, pigboat, sub, U-boat', 'bow', 'sundial', 'rugby ball', 'cockroach, roach', 'agama', 'manhole cover', 'oystercatcher, oyster catcher', 'bee', 
                              'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'shovel', 'cliff, drop, drop-off', 'puffer, pufferfish, blowfish, globefish', 
                              "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'beaker', 'flagpole, flagstaff', 'viaduct', 
                              'American alligator, Alligator mississipiensis', 'breastplate, aegis, egis', 'digital clock', 'balloon', 'cheeseburger', 'pug, pug-dog', 
                              'German shepherd, German shepherd dog, German police dog, alsatian', 'reel', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 
                              'wood rabbit, cottontail, cottontail rabbit', 'torch', 'centipede', 'airliner', 'doormat, welcome mat', 'cowboy boot', 'jeep, landrover', 'guacamole', 
                              'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'porcupine, hedgehog', 'bikini, two-piece', 'cabbage butterfly', 'weevil', 'snail', 
                              'pretzel', 'marmot', 'banana', 'water tower', 'quill, quill pen', 'eft', 'ant, emmet, pismire', 'teddy, teddy bear', 'parachute, chute', 
                              'marimba, xylophone', "spider web, spider's web", 'studio couch, day bed', 'forklift', 'custard apple', 'bubble', 'jay', 'basketball', 'flamingo', 
                              'schooner', 'drumstick', 'unicycle, monocycle', 'envelope', 'revolver, six-gun, six-shooter', 'cello, violoncello', 'Chihuahua', 'grasshopper, hopper', 
                              'lemon', 'castle', 'chest', 'mantis, mantid', 'sax, saxophone', 'broom', 'goose', 'parking meter', 'armadillo', 'flatworm, platyhelminth', 'bell pepper', 
                              'sleeping bag', 'rhinoceros beetle', 'box turtle, box tortoise', 'starfish, sea star', 'harvestman, daddy longlegs, Phalangium opilio', 'feather boa, boa', 
                              'tarantula', 'mongoose', 'bow tie, bow-tie, bowtie', 'suspension bridge', 'accordion, piano accordion, squeeze box', 'junco, snowbird', 'pomegranate', 
                              'mitten', 'maraca', 'pelican', 'washer, automatic washer, washing machine', 'snowplow, snowplough', 'spatula', 
                              'bald eagle, American eagle, Haliaeetus leucocephalus', 'toaster', 'racket, racquet', 'mosque', 'baboon', 'tricycle, trike, velocipede', 'sandal', 
                              'rapeseed', 'ballplayer, baseball player', 'stethoscope', 'sewing machine', 'saltshaker, salt shaker', 'canoe', 'sea anemone, anemone', 'cradle', 
                              'barrow, garden cart, lawn cart, wheelbarrow', 'capuchin, ringtail, Cebus capucinus', 'dumbbell', 'acorn', 'snowmobile', 
                              'walking stick, walkingstick, stick insect', 'steam locomotive']


imagenet_A_class_name_first = ['bullfrog', 'ambulance', 'school bus', 'lorikeet', 'African chameleon', 'golfcart', 'chain', 'lion', 'organ', 'ocarina', 'Christmas stocking', 
                               'African elephant', 'volleyball', 'dragonfly', 'vulture', 'nail', 'sulphur-crested cockatoo', 'toucan', 'common iguana', 'go-kart', 'academic gown', 
                               'goblet', 'golden retriever', 'umbrella', 'drake', 'wreck', 'volcano', 'iron', 'goldfinch', 'sea lion', 'leafhopper', 'syringe', 'hummingbird', 
                               'puck', 'skunk', 'red fox', 'barn', 'candle', "jack-o'-lantern", 'pool table', 'kimono', 'cucumber', 'jellyfish', 'scorpion', 'banjo', 'Persian cat', 
                               'ladybug', 'beacon', 'carbonara', 'obelisk', 'hotdog', 'garter snake', 'robin', 'mushroom', 'corn', 'lighter', 'American egret', 'fountain', 
                               'piggy bank', 'stingray', 'wine bottle', 'crayfish', 'garbage truck', 'acoustic guitar', 'balance beam', 'apron', 'American black bear', 'grand piano', 
                               'broccoli', 'soap dispenser', 'dial telephone', 'limousine', 'fly', 'rocking chair', 'koala', 'lycaenid', 'fox squirrel', 'lynx', 'hermit crab', 
                               'bison', 'mask', 'Rottweiler', 'submarine', 'bow', 'sundial', 'rugby ball', 'cockroach', 'agama', 'manhole cover', 'oystercatcher', 'bee', 'monarch', 
                               'shovel', 'cliff', 'puffer', "yellow lady's slipper", 'beaker', 'flagpole', 'viaduct', 'American alligator', 'breastplate', 'digital clock', 'balloon', 
                               'cheeseburger', 'pug', 'German shepherd', 'reel', 'tank', 'wood rabbit', 'torch', 'centipede', 'airliner', 'doormat', 'cowboy boot', 'jeep', 'guacamole', 
                               'hand blower', 'porcupine', 'bikini', 'cabbage butterfly', 'weevil', 'snail', 'pretzel', 'marmot', 'banana', 'water tower', 'quill', 'eft', 'ant', 
                               'teddy', 'parachute', 'marimba', 'spider web', 'studio couch', 'forklift', 'custard apple', 'bubble', 'jay', 'basketball', 'flamingo', 'schooner', 
                               'drumstick', 'unicycle', 'envelope', 'revolver', 'cello', 'Chihuahua', 'grasshopper', 'lemon', 'castle', 'chest', 'mantis', 'sax', 'broom', 'goose', 
                               'parking meter', 'armadillo', 'flatworm', 'bell pepper', 'sleeping bag', 'rhinoceros beetle', 'box turtle', 'starfish', 'harvestman', 'feather boa', 
                               'tarantula', 'mongoose', 'bow tie', 'suspension bridge', 'accordion', 'junco', 'pomegranate', 'mitten', 'maraca', 'pelican', 'washer', 'snowplow', 
                               'spatula', 'bald eagle', 'toaster', 'racket', 'mosque', 'baboon', 'tricycle', 'sandal', 'rapeseed', 'ballplayer', 'stethoscope', 'sewing machine', 
                               'saltshaker', 'canoe', 'sea anemone', 'cradle', 'barrow', 'capuchin', 'dumbbell', 'acorn', 'snowmobile', 'walking stick', 'steam locomotive']

imagenet_A_claas_names = imagenet_A_class_name_first

def get_imagenet_A_prompt():
    prompt = [f'a photo of a {class_name}' for class_name in imagenet_A_claas_names]
    return prompt







imagenet_1k_class_names = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 
                          'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle', 
                          'vulture', 'great grey owl', 'European fire salamander', 'common newt', 'eft', 'spotted salamander', 'axolotl', 'bullfrog', 'tree frog', 
                          'tailed frog', 'loggerhead', 'leatherback turtle', 'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana', 'American chameleon', 
                          'whiptail', 'agama', 'frilled lizard', 'alligator lizard', 'Gila monster', 'green lizard', 'African chameleon', 'Komodo dragon', 'African crocodile', 
                          'American alligator', 'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake', 'green snake', 'king snake', 'garter snake', 'water snake', 
                          'vine snake', 'night snake', 'boa constrictor', 'rock python', 'Indian cobra', 'green mamba', 'sea snake', 'horned viper', 'diamondback', 'sidewinder', 
                          'trilobite', 'harvestman', 'scorpion', 'black and gold garden spider', 'barn spider', 'garden spider', 'black widow', 'tarantula', 'wolf spider', 'tick', 
                          'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie chicken', 'peacock', 'quail', 'partridge', 'African grey', 'macaw', 
                          'sulphur crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red breasted merganser', 'goose', 
                          'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode', 'conch', 
                          'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab', 'king crab', 'American lobster', 'spiny lobster', 
                          'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron', 'American egret', 'bittern', 'crane', 
                          'limpkin', 'European gallinule', 'American coot', 'bustard', 'ruddy turnstone', 'red backed sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'pelican', 
                          'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih Tzu', 
                          'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black and tan coonhound', 
                          'Walker hound', 'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound', 'Norwegian elkhound', 'otterhound', 
                          'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 
                          'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 
                          'Airedale', 'cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier', 
                          'Tibetan terrier', 'silky terrier', 'soft coated wheaten terrier', 'West Highland white terrier', 'Lhasa', 'flat coated retriever', 'curly coated retriever', 
                          'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short haired pointer', 'vizsla', 'English setter', 'Irish setter', 'Gordon setter', 
                          'Brittany spaniel', 'clumber', 'English springer', 'Welsh springer spaniel', 'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 
                          'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog', 'Shetland sheepdog', 'collie', 'Border collie', 'Bouvier des Flandres', 
                          'Rottweiler', 'German shepherd', 'Doberman', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 
                          'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute', 'Siberian husky', 'dalmatian', 'affenpinscher', 
                          'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'toy poodle', 
                          'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf', 'white wolf', 'red wolf', 'coyote', 'dingo', 'dhole', 'African hunting dog', 'hyena', 
                          'red fox', 'kit fox', 'Arctic fox', 'grey fox', 'tabby', 'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian cat', 'cougar', 'lynx', 'leopard', 'snow leopard', 
                          'jaguar', 'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear', 'ice bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 
                          'long horned beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking stick', 'cockroach', 
                          'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage butterfly', 'sulphur butterfly', 'lycaenid', 
                          'starfish', 'sea urchin', 'sea cucumber', 'wood rabbit', 'hare', 'Angora', 'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'sorrel', 
                          'zebra', 'hog', 'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn', 'ibex', 'hartebeest', 'impala', 'gazelle', 'Arabian camel', 
                          'llama', 'weasel', 'mink', 'polecat', 'black footed ferret', 'otter', 'skunk', 'badger', 'armadillo', 'three toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 
                          'gibbon', 'siamang', 'guenon', 'patas', 'baboon', 'macaque', 'langur', 'colobus', 'proboscis monkey', 'marmoset', 'capuchin', 'howler monkey', 'titi', 'spider monkey', 
                          'squirrel monkey', 'Madagascar cat', 'indri', 'Indian elephant', 'African elephant', 'lesser panda', 'giant panda', 'barracouta', 'eel', 'coho', 'rock beauty', 
                          'anemone fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus', 'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier', 'airliner', 'airship', 
                          'altar', 'ambulance', 'amphibian', 'analog clock', 'apiary', 'apron', 'ashcan', 'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon', 'ballpoint', 'Band Aid', 
                          'banjo', 'bannister', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'barrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap', 
                          'bath towel', 'bathtub', 'beach wagon', 'beacon', 'beaker', 'bearskin', 'beer bottle', 'beer glass', 'bell cote', 'bib', 'bicycle built for two', 'bikini', 'binder', 
                          'binoculars', 'birdhouse', 'boathouse', 'bobsled', 'bolo tie', 'bonnet', 'bookcase', 'bookshop', 'bottlecap', 'bow', 'bow tie', 'brass', 'brassiere', 'breakwater', 
                          'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest', 'bullet train', 'butcher shop', 'cab', 'caldron', 'candle', 'cannon', 'canoe', 'can opener', 'cardigan', 
                          'car mirror', 'carousel', "carpenter's kit", 'carton', 'car wheel', 'cash machine', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello', 
                          'cellular telephone', 'chain', 'chainlink fence', 'chain mail', 'chain saw', 'chest', 'chiffonier', 'chime', 'china cabinet', 'Christmas stocking', 'church', 'cinema', 
                          'cleaver', 'cliff dwelling', 'cloak', 'clog', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil', 'combination lock', 'computer keyboard', 'confectionery', 
                          'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib', 'Crock Pot', 'croquet ball', 
                          'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'dial telephone', 'diaper', 'digital clock', 'digital watch', 'dining table', 'dishrag', 'dishwasher', 'disk brake', 
                          'dock', 'dogsled', 'dome', 'doormat', 'drilling platform', 'drum', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar', 'electric locomotive', 
                          'entertainment center', 'envelope', 'espresso maker', 'face powder', 'feather boa', 'file', 'fireboat', 'fire engine', 'fire screen', 'flagpole', 'flute', 'folding chair', 
                          'football helmet', 'forklift', 'fountain', 'fountain pen', 'four poster', 'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck', 'gasmask', 'gas pump', 
                          'goblet', 'go kart', 'golf ball', 'golfcart', 'gondola', 'gong', 'gown', 'grand piano', 'greenhouse', 'grille', 'grocery store', 'guillotine', 'hair slide', 'hair spray', 
                          'half track', 'hammer', 'hamper', 'hand blower', 'hand held computer', 'handkerchief', 'hard disc', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'home theater', 
                          'honeycomb', 'hook', 'hoopskirt', 'horizontal bar', 'horse cart', 'hourglass', 'iPod', 'iron', "jack o' lantern", 'jean', 'jeep', 'jersey', 'jigsaw puzzle', 'jinrikisha', 
                          'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade', 'laptop', 'lawn mower', 'lens cap', 'letter opener', 'library', 'lifeboat', 'lighter', 'limousine', 
                          'liner', 'lipstick', 'Loafer', 'lotion', 'loudspeaker', 'loupe', 'lumbermill', 'magnetic compass', 'mailbag', 'mailbox', 'maillot', 'maillot', 'manhole cover', 'maraca', 
                          'marimba', 'mask', 'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine chest', 'megalith', 'microphone', 'microwave', 'military uniform', 'milk can', 'minibus', 
                          'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 
                          'motor scooter', 'mountain bike', 'mountain tent', 'mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook', 'obelisk', 'oboe', 
                          'ocarina', 'odometer', 'oil filter', 'organ', 'oscilloscope', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 
                          'palace', 'panpipe', 'paper towel', 'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car', 'patio', 'pay phone', 'pedestal', 'pencil box', 
                          'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'pick', 'pickelhaube', 'picket fence', 'pickup', 'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping pong ball', 
                          'pinwheel', 'pirate', 'pitcher', 'plane', 'planetarium', 'plastic bag', 'plate rack', 'plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho', 'pool table', 
                          'pop bottle', 'pot', "potter's wheel", 'power drill', 'prayer rug', 'printer', 'prison', 'projectile', 'projector', 'puck', 'punching bag', 'purse', 'quill', 'quilt', 'racer', 
                          'racket', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'recreational vehicle', 'reel', 'reflex camera', 'refrigerator', 'remote control', 'restaurant', 'revolver', 
                          'rifle', 'rocking chair', 'rotisserie', 'rubber eraser', 'rugby ball', 'rule', 'running shoe', 'safe', 'safety pin', 'saltshaker', 'sandal', 'sarong', 'sax', 'scabbard', 'scale', 
                          'school bus', 'schooner', 'scoreboard', 'screen', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe shop', 'shoji', 'shopping basket', 'shopping cart', 
                          'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule', 'sliding door', 'slot', 'snorkel', 'snowmobile', 'snowplow', 'soap dispenser', 
                          'soccer ball', 'sock', 'solar dish', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', 'spider web', 'spindle', 'sports car', 
                          'spotlight', 'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch', 'stove', 'strainer', 'streetcar', 'stretcher', 
                          'studio couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension bridge', 'swab', 'sweatshirt', 'swimming trunks', 'swing', 'switch', 
                          'syringe', 'table lamp', 'tank', 'tape player', 'teapot', 'teddy', 'television', 'tennis ball', 'thatch', 'theater curtain', 'thimble', 'thresher', 'throne', 'tile roof', 'toaster', 
                          'tobacco shop', 'toilet seat', 'torch', 'totem pole', 'tow truck', 'toyshop', 'tractor', 'trailer truck', 'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch', 
                          'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle', 'upright', 'vacuum', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 
                          'violin', 'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'warplane', 'washbasin', 'washer', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 
                          'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool', 'worm fence', 'wreck', 'yawl', 'yurt', 'web site', 'comic book', 
                          'crossword puzzle', 'street sign', 'traffic light', 'book jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'ice lolly', 'French loaf', 'bagel', 
                          'pretzel', 'cheeseburger', 'hotdog', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 
                          'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 
                          'hay', 'carbonara', 'chocolate sauce', 'dough', 'meat loaf', 'pizza', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff', 'coral reef', 'geyser', 
                          'lakeside', 'promontory', 'sandbar', 'seashore', 'valley', 'volcano', 'ballplayer', 'groom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'hip', 
                          'buckeye', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar', 'hen of the woods', 'bolete', 'ear', 'toilet tissue']

def get_imagenet_1k_prompt():
    prompt = [f'a photo of a {class_name}' for class_name in imagenet_1k_class_names]
    return prompt










imagenet_R_class_names = ['goldfish', 'great white shark', 'hammerhead', 'stingray', 'hen', 'ostrich', 'goldfinch', 'junco', 'bald eagle', 'vulture', 'newt', 'axolotl', 
                          'tree frog', 'iguana', 'African chameleon', 'cobra', 'scorpion', 'tarantula', 'centipede', 'peacock', 'lorikeet', 'hummingbird', 'toucan', 'duck', 
                          'goose', 'black swan', 'koala', 'jellyfish', 'snail', 'lobster', 'hermit crab', 'flamingo', 'american egret', 'pelican', 'king penguin', 'grey whale', 
                          'killer whale', 'sea lion', 'chihuahua', 'shih tzu', 'afghan hound', 'basset hound', 'beagle', 'bloodhound', 'italian greyhound', 'whippet', 
                          'weimaraner', 'yorkshire terrier', 'boston terrier', 'scottish terrier', 'west highland white terrier', 'golden retriever', 'labrador retriever', 
                          'cocker spaniels', 'collie', 'border collie', 'rottweiler', 'german shepherd dog', 'boxer', 'french bulldog', 'saint bernard', 'husky', 'dalmatian', 
                          'pug', 'pomeranian', 'chow chow', 'pembroke welsh corgi', 'toy poodle', 'standard poodle', 'timber wolf', 'hyena', 'red fox', 'tabby cat', 'leopard', 
                          'snow leopard', 'lion', 'tiger', 'cheetah', 'polar bear', 'meerkat', 'ladybug', 'fly', 'bee', 'ant', 'grasshopper', 'cockroach', 'mantis', 
                          'dragonfly', 'monarch butterfly', 'starfish', 'wood rabbit', 'porcupine', 'fox squirrel', 'beaver', 'guinea pig', 'zebra', 'pig', 'hippopotamus', 
                          'bison', 'gazelle', 'llama', 'skunk', 'badger', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'baboon', 'panda', 'eel', 'clown fish', 'puffer fish', 
                          'accordion', 'ambulance', 'assault rifle', 'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 'beer glass', 'binoculars', 
                          'birdhouse', 'bow tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 'castle', 'mobile phone', 'cowboy hat', 
                          'electric guitar', 'fire engine', 'flute', 'gasmask', 'grand piano', 'guillotine', 'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 
                          'lab coat', 'lawn mower', 'lipstick', 'mailbox', 'missile', 'mitten', 'parachute', 'pickup truck', 'pirate ship', 'revolver', 'rugby ball', 'sandal', 
                          'saxophone', 'school bus', 'schooner', 'shield', 'soccer ball', 'space shuttle', 'spider web', 'steam locomotive', 'scarf', 'submarine', 'tank', 
                          'tennis ball', 'tractor', 'trombone', 'vase', 'violin', 'military aircraft', 'wine bottle', 'ice cream', 'bagel', 'pretzel', 'cheeseburger', 
                          'hotdog', 'cabbage', 'broccoli', 'cucumber', 'bell pepper', 'mushroom', 'Granny Smith', 'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate', 
                          'pizza', 'burrito', 'espresso', 'volcano', 'baseball player', 'scuba diver', 'acorn']


def get_imagenet_R_prompt():
    prompt = [f'a photo of a {class_name}' for class_name in imagenet_R_class_names]
    return prompt