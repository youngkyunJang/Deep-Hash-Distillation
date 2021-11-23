from config import *

class Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
        image = Image.open(img_name)

        if self.NB_CLS != None:
            if len(self.file_list[idx])>2:
                label = [int(self.file_list[idx][i]) for i in range(1,self.NB_CLS+1)]
                label = T.FloatTensor(label)
            else:
                label = int(self.file_list[idx][1])
            return transforms.ToTensor()(image), label
        else:
            return transforms.ToTensor()(image)
