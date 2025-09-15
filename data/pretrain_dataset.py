from data.utils import pre_caption
import json
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def read_json(file):
    f = open(file, "r", encoding="utf-8").read()
    return json.loads(f)

def concat_json(*files):
    json_contents = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            json_contents.append(json.load(f))

    merged_json = {}
    for json_content in json_contents:
        merged_json.update(json_content)

    return merged_json

class pretrain_natural(Dataset):
    def __init__(self, config, natural_id_label, all_id_info, transform, task_i_list, memory_item_ids=[[], []]):
        self.image_path = config['train_image_root']
        self.max_words = config['max_words']
        self.transform = transform
        self.data_list = []
        self.natural_id_label = natural_id_label
        self.all_id_info = all_id_info

        for task_i in task_i_list:
            for item_id, info in self.natural_id_label[task_i].items():
                self.data_list.append((item_id, info["query-image"][1],info["target-image"][1], info["query-text"]))

        for i, item_id in enumerate(memory_item_ids[0]):
            label = memory_item_ids[1][i]
            self.data_list.append((item_id, self.all_id_info[item_id]["query-image"][1], self.all_id_info[item_id]["target-image"][1], self.all_id_info[item_id]["query-text"]))

        print("Total Training Pairs: {}".format(len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item_id, source_img, target_img, modification_text = self.data_list[index]

        source_image_path = "{}/{}".format(self.image_path, source_img)
        target_image_path = "{}/{}".format(self.image_path, target_img)
        try:
            s_image = Image.open(source_image_path).convert('RGB')
            t_image = Image.open(target_image_path).convert('RGB')
        except:
            print(source_image_path, target_image_path)

        s_image = self.transform(s_image)
        t_image = self.transform(t_image)

        return item_id, s_image, t_image, modification_text


class test_natural(Dataset):
    def __init__(self, config, test_file, transform, task_i_list):
        self.image_path = config['test_image_root']
        self.test_files = test_file

        print("Loading test data from {}".format(test_file))

        self.task_i_list = task_i_list
        self.transform = transform
        self.max_words = config['max_words']
        self.data_list = []

        # Load the data for the given tasks
        for i, task in enumerate(task_i_list):
            self.all_id_info = read_json(self.test_files)

            for item_id, info in self.all_id_info[task].items():
                super_entity = info["super_entity"]
                if super_entity not in task_i_list:
                    continue
                else:
                    self.data_list.append(
                        (item_id, info["query-image"][1],info["target-image"][1], info["query-text"], info["super_entity"])
                    )

        self.dataset_len = len(self.data_list)
        print("Total Test Pairs: {}".format(len(self.data_list)))

        self.source_img = []
        self.target_img = []
        self.modification_text = []
        self.fusion2img = {}
        for id, item in enumerate(self.data_list):
            self.source_img.append(item[1])
            self.target_img.append(item[2])
            self.modification_text.append(pre_caption(item[3], self.max_words))  
            self.fusion2img[id] = id

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        item_id, source_img, target_img, modification_text,_ = self.data_list[index]

        source_image_path = "{}/{}".format(self.image_path, source_img)
        target_image_path = "{}/{}".format(self.image_path, target_img)

        try:
            s_image = Image.open(source_image_path).convert('RGB')
            t_image = Image.open(target_image_path).convert('RGB')
        except:
            print(source_image_path, target_image_path)

        s_image = self.transform(s_image)
        t_image = self.transform(t_image)

        return item_id, s_image, t_image, modification_text


class pretrain_fashion(Dataset):
    def __init__(self, config, fashion_id_label, all_id_info, transform, task_i_list, memory_item_ids=[[], []]):
        self.image_path = config['train_image_root']
        self.max_words = config['max_words']
        self.transform = transform
        self.data_list = []
        self.fashion_id_label = fashion_id_label
        self.all_id_info = all_id_info

        for task_i in task_i_list:
            for item_id, info in self.fashion_id_label[task_i].items():
                self.data_list.append(
                    (item_id, info["source_img"],info["target_img"], info["modification_text"]))

        for i, item_id in enumerate(memory_item_ids[0]):
            label = memory_item_ids[1][i]
            self.data_list.append(
                (item_id, self.all_id_info[item_id]["source_img"], self.all_id_info[item_id]["target_img"], self.all_id_info[item_id]["modification_text"]))

        print("Total Training Pairs: {}".format(len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item_id, source_img, target_img, modification_text = self.data_list[index]

        if item_id.startswith('shoes'):
            image_path = self.image_path + '/Shoes/'
        else:
            image_path = self.image_path + '/Fashion-IQ/images/'

        source_image_path = "{}/{}".format(image_path, source_img)
        target_image_path = "{}/{}".format(image_path, target_img)

        try:
            s_image = Image.open(source_image_path).convert('RGB')
            t_image = Image.open(target_image_path).convert('RGB')
        except:
            print(source_image_path, target_image_path)

        s_image = self.transform(s_image)
        t_image = self.transform(t_image)

        return item_id, s_image, t_image, modification_text


class test_fashion(Dataset):
    def __init__(self, config, test_file, transform, task_i_list):
        self.image_path = config['test_image_root']
        self.test_files = test_file

        print("Loading test data from {}".format(test_file))

        self.task_i_list = task_i_list
        self.transform = transform
        self.max_words = config['max_words']
        self.data_list = []

        # Load the data for the given tasks
        for i, task in enumerate(task_i_list):
            self.all_id_info = read_json(self.test_files)

            for item_id, info in self.all_id_info[task].items():
                """
                ['dress', 'shirt', 'toptee', 'shoes']
                """
                if item_id.startswith('dress'):
                    fashion_name = 'dress'
                elif item_id.startswith('shirt'):
                    fashion_name = 'shirt'
                elif item_id.startswith('toptee'):
                    fashion_name = 'toptee'
                elif item_id.startswith('shoes'):
                    fashion_name = 'shoes'

                if any(fashion_name.startswith(task) for task in self.task_i_list):
                    self.data_list.append(
                        (item_id, info["source_img"], info["target_img"], info["modification_text"])
                    )

        self.dataset_len = len(self.data_list)
        print("Total Test Pairs: {}".format(len(self.data_list)))

        self.fusion2img = {}
        for id, item in enumerate(self.data_list):
            self.fusion2img[id] = id

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        item_id, source_img, target_img, modification_text = self.data_list[index]

        if item_id.startswith('shoes'):
            image_path = self.image_path + '/Shoes/'
        else:
            image_path = self.image_path + '/Fashion-IQ/images/'

        source_image_path = "{}/{}".format(image_path, source_img)
        target_image_path = "{}/{}".format(image_path, target_img)

        try:
            s_image = Image.open(source_image_path).convert('RGB')
            t_image = Image.open(target_image_path).convert('RGB')
        except:
            print(source_image_path, target_image_path)

        s_image = self.transform(s_image)
        t_image = self.transform(t_image)

        return item_id, s_image, t_image, modification_text
